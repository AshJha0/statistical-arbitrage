package com.quant.statarb;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Pairs backtest — state machine and accounting (PINNED).
 *
 * <p>Timing convention (pinned, no look-ahead): the signal computed from the
 * day-t CLOSE is executed at the day t+1 CLOSE at day t+1 prices. Daily P&amp;L
 * on day t marks the position held since the day t-1 close:
 *
 * <pre>
 *     pnl_t  = q_y * (y_t - y_{t-1}) + q_x * (x_t - x_{t-1})
 *     cost_t = (cost_bps + slippage_bps) / 1e4 * traded_notional_t
 *     E_t    = E_{t-1} + pnl_t - cost_t,    E_{-1} := capital
 * </pre>
 *
 * so the accounting identity {@code E_{T-1} - capital == sum(pnl) - sum(costs)}
 * holds to rounding (tested at 1e-10).
 *
 * <p>State machine (states FLAT / LONG spread / SHORT spread, plus at most one
 * pending order):
 * <ul>
 *   <li>FLAT, signal day t (and {@code t >= e + cooldownDays} for the latest
 *       stopped exit executed on day e): {@code entry_z <= z_t < stop_z} -&gt;
 *       pending SHORT entry; {@code -stop_z < z_t <= -entry_z} -&gt; pending
 *       LONG entry ({@code |z_t| >= stop_z} while flat: no entry, pinned; NaN z:
 *       no action).</li>
 *   <li>In position, signal day t, first match wins: (1) open trade P&amp;L
 *       (net of costs so far) {@code <= -pnlStopFrac * capital} -&gt; STOP;
 *       (2) {@code |z_t| >= stop_z} -&gt; STOP; (3)
 *       {@code t - entryDay >= maxHoldDays} -&gt; exit (not a stop); (4)
 *       {@code |z_t| <= exit_z} -&gt; exit. Rules 1 and 3 do not read z, so they
 *       fire on a NaN z. A limit of 0 is off.</li>
 *   <li>Execution day t+1 close, dir = +1 for LONG spread (long y / short x),
 *       -1 for SHORT. Sizing {@code beta_hedged} (default):
 *       {@code u = capital / (y + beta*x)}, {@code q_y = dir*u},
 *       {@code q_x = -dir*beta*u}; {@code equal_dollar}:
 *       {@code q_y = dir*(capital/2)/y}, {@code q_x = -dir*(capital/2)/x}. Gross
 *       notional is capital at entry either way. Quantities are then held fixed
 *       (no daily rebalance).</li>
 *   <li>Costs are charged on traded notional {@code |dq_y| y + |dq_x| x} at the
 *       execution close, both entry and exit.</li>
 *   <li>A position still open at the final day is force-closed at the final
 *       close (a normal closed trade, not a stop). A pending entry at the end
 *       of data is DISCARDED (never executed-and-force-closed at the same
 *       close).</li>
 *   <li>Metrics: {@code ann_return = (1+total)^(252/N) - 1}, pinned to -1.0
 *       when the equity curve ends at or below zero (total_return &le; -1).</li>
 * </ul>
 */
public final class Backtest {

    private static final int FLAT = 0;

    /**
     * Nudge added before flooring {@code T * insampleFrac} in
     * {@link #walkForward} so the split is the intended integer despite
     * binary-float products ({@code 700 * 0.7 == 489.99999999999994}). Pinned.
     */
    public static final double SPLIT_EPS = 1.0e-9;

    private Backtest() {
    }

    /** Accepted {@link Config#sizing()} name (default). */
    public static final String BETA_HEDGED = "beta_hedged";
    /** Accepted {@link Config#sizing()} name. */
    public static final String EQUAL_DOLLAR = "equal_dollar";

    /** Largest |z| a W-point sample can give its own latest point (ddof=1). */
    public static double maxAbsZscore(int window) {
        return (window - 1) / Math.sqrt(window);
    }

    /**
     * Signal / cost / risk parameters for the pairs backtest, validated on
     * construction.
     *
     * @param window       rolling z-score window W (&ge; 2)
     * @param entryZ       enter when |z| &ge; entryZ (and &lt; stopZ)
     * @param exitZ        exit when |z| &le; exitZ
     * @param stopZ        stop-loss when |z| &ge; stopZ
     * @param costBps      commission, basis points of traded notional
     * @param slippageBps  slippage, basis points of traded notional
     * @param capital      gross capital at entry
     * @param sizing       "beta_hedged" or "equal_dollar"
     * @param pnlStopFrac  stop when open trade P&amp;L &le; -frac*capital; 0 = off
     * @param maxHoldDays  exit when held this many days; 0 = off
     * @param cooldownDays no entry signal for this many days after a stopped
     *                     exit; 0 = off
     */
    public record Config(int window, double entryZ, double exitZ, double stopZ,
                         double costBps, double slippageBps, double capital,
                         String sizing, double pnlStopFrac, int maxHoldDays,
                         int cooldownDays) {

        /**
         * Validates the pinned parameter constraints.
         *
         * @throws IllegalArgumentException when window &lt; 2, any threshold or
         *     cost is NaN/infinite, the ordering is violated, costs are negative,
         *     capital is not positive and finite, sizing is unknown, a risk
         *     limit is negative or non-finite, or the config is dead (entry_z
         *     unreachable, or stop_z unreachable without a P&amp;L stop)
         */
        public Config {
            if (window < 2) {
                throw new IllegalArgumentException("window must be an integer >= 2");
            }
            if (!(Double.isFinite(entryZ) && Double.isFinite(exitZ) && Double.isFinite(stopZ)
                    && Double.isFinite(costBps) && Double.isFinite(slippageBps))) {
                throw new IllegalArgumentException("thresholds and costs must be finite");
            }
            if (!(0.0 <= exitZ && exitZ < entryZ && entryZ < stopZ)) {
                throw new IllegalArgumentException("require 0 <= exit_z < entry_z < stop_z");
            }
            if (costBps < 0.0 || slippageBps < 0.0) {
                throw new IllegalArgumentException("costs must be >= 0");
            }
            if (!(capital > 0.0 && Double.isFinite(capital))) {
                throw new IllegalArgumentException("capital must be positive and finite");
            }
            if (!(BETA_HEDGED.equals(sizing) || EQUAL_DOLLAR.equals(sizing))) {
                throw new IllegalArgumentException(
                        "sizing must be 'beta_hedged' or 'equal_dollar'");
            }
            if (!(Double.isFinite(pnlStopFrac) && pnlStopFrac >= 0.0)) {
                throw new IllegalArgumentException("pnl_stop_frac must be finite and >= 0");
            }
            if (maxHoldDays < 0) {
                throw new IllegalArgumentException("max_hold_days must be an integer >= 0");
            }
            if (cooldownDays < 0) {
                throw new IllegalArgumentException("cooldown_days must be an integer >= 0");
            }
            double reach = maxAbsZscore(window);
            if (reach <= entryZ) {
                throw new IllegalArgumentException(
                        "entry_z unreachable: window " + window + " caps |z| at " + reach);
            }
            if (reach <= stopZ && !(pnlStopFrac > 0.0)) {
                throw new IllegalArgumentException(
                        "stop_z unreachable: window " + window + " caps |z| at " + reach
                                + "; allowed only with pnl_stop_frac > 0");
            }
        }

        /** Beta-hedged sizing, all risk limits off. */
        public Config(int window, double entryZ, double exitZ, double stopZ,
                      double costBps, double slippageBps, double capital) {
            this(window, entryZ, exitZ, stopZ, costBps, slippageBps, capital,
                    BETA_HEDGED, 0.0, 0, 0);
        }

        /** Copy with another sizing mode. */
        public Config withSizing(String newSizing) {
            return new Config(window, entryZ, exitZ, stopZ, costBps, slippageBps, capital,
                    newSizing, pnlStopFrac, maxHoldDays, cooldownDays);
        }

        /** Copy with other risk limits. */
        public Config withLimits(double newPnlStopFrac, int newMaxHoldDays,
                                 int newCooldownDays) {
            return new Config(window, entryZ, exitZ, stopZ, costBps, slippageBps, capital,
                    sizing, newPnlStopFrac, newMaxHoldDays, newCooldownDays);
        }

        /**
         * Build from a {@link DataIo#loadConfig} map; absent risk limits are
         * off, sizing is beta_hedged.
         */
        public static Config fromMap(Map<String, Double> c) {
            return new Config(c.get("window").intValue(), c.get("entry_z"), c.get("exit_z"),
                    c.get("stop_z"), c.get("cost_bps"), c.get("slippage_bps"),
                    c.get("capital"), BETA_HEDGED,
                    c.getOrDefault("pnl_stop_frac", 0.0),
                    c.getOrDefault("max_hold_days", 0.0).intValue(),
                    c.getOrDefault("cooldown_days", 0.0).intValue());
        }
    }

    /**
     * One closed round trip (days are execution-day indices).
     *
     * @param entryDay  execution day of the entry
     * @param exitDay   execution day of the exit
     * @param direction +1 long spread, -1 short spread
     * @param pnl       net of entry and exit costs
     * @param stopped   true when closed by the stop-loss rule
     */
    public record Trade(int entryDay, int exitDay, int direction, double pnl,
                        boolean stopped) {

        /** Days held: {@code exitDay - entryDay}. */
        public int holdingDays() {
            return exitDay - entryDay;
        }
    }

    /**
     * Full backtest output: paths, trades, and pinned metrics.
     *
     * @param equity    equity curve E_t, length T
     * @param dailyPnl  mark-to-market P&amp;L per day
     * @param dailyCost transaction cost charged per day
     * @param zscore    the rolling z-score used as the signal
     * @param trades    closed round trips in chronological order
     * @param metrics   pinned metric map (total_return, ann_return, sharpe,
     *                  max_drawdown, n_trades, win_rate, avg_holding_days,
     *                  turnover)
     */
    public record Result(double[] equity, double[] dailyPnl, double[] dailyCost,
                         double[] zscore, List<Trade> trades, Map<String, Double> metrics) {

        /** Number of closed trades that ended via the stop-loss rule. */
        public int nStops() {
            int c = 0;
            for (Trade t : trades) {
                if (t.stopped()) {
                    c++;
                }
            }
            return c;
        }
    }

    /**
     * In-sample vs out-of-sample comparison with a single hedge estimate.
     *
     * @param alpha    hedge intercept estimated in-sample
     * @param beta     hedge ratio estimated in-sample
     * @param insample backtest of the in-sample segment
     * @param oos      backtest of the out-of-sample segment (same hedge)
     * @param split    first out-of-sample day index
     */
    public record WalkForwardResult(double alpha, double beta, Result insample,
                                    Result oos, int split) {
    }

    /**
     * Compute the pinned metric set from an equity curve and trade list.
     *
     * <p>Formulas (N days): {@code daily_ret_t = (E_t - E_{t-1}) / capital};
     * {@code total_return = (E_{N-1} - capital)/capital};
     * {@code ann_return = (1+total_return)^(252/N) - 1};
     * {@code sharpe = mean(daily_ret)/std(daily_ret, ddof=1) * sqrt(252)}
     * (0 when std = 0 or N &lt; 2);
     * {@code max_drawdown = max_t (peak_t - E_t)/peak_t} with peak the running
     * max of E (positive fraction); win_rate = wins/n_trades (0 if none);
     * avg_holding_days = mean(exitDay - entryDay) (0 if none).
     * {@code ann_return} is pinned to -1.0 when {@code total_return <= -1}
     * (the power is undefined for a non-positive base).
     *
     * @throws IllegalArgumentException on an empty or non-finite equity curve
     *     or a non-positive / non-finite capital
     */
    public static Map<String, Double> computeMetrics(double[] equity, double capital,
                                                     List<Trade> trades) {
        int n = equity.length;
        if (n == 0) {
            throw new IllegalArgumentException("equity curve must be a non-empty 1-D array");
        }
        if (!LinAlg.allFinite(equity)) {
            throw new IllegalArgumentException("equity curve must be finite (no NaN/inf)");
        }
        if (!(capital > 0.0 && Double.isFinite(capital))) {
            throw new IllegalArgumentException("capital must be positive and finite");
        }
        double totalReturn = (equity[n - 1] - capital) / capital;
        double annReturn = totalReturn <= -1.0
                ? -1.0
                : Math.pow(1.0 + totalReturn, 252.0 / n) - 1.0;
        double[] dailyRet = new double[n];
        double prev = capital;
        for (int t = 0; t < n; t++) {
            dailyRet[t] = (equity[t] - prev) / capital;
            prev = equity[t];
        }
        double sharpe = 0.0;
        if (n >= 2) {
            double m = LinAlg.mean(dailyRet);
            double[] dev = new double[n];
            for (int t = 0; t < n; t++) {
                dev[t] = dailyRet[t] - m;
            }
            double sd = Math.sqrt(LinAlg.dot(dev, dev) / (n - 1));
            if (sd > 0.0) {
                sharpe = m / sd * Math.sqrt(252.0);
            }
        }
        double peak = Double.NEGATIVE_INFINITY;
        double maxDd = Double.NEGATIVE_INFINITY;
        for (int t = 0; t < n; t++) {
            peak = Math.max(peak, equity[t]);
            maxDd = Math.max(maxDd, (peak - equity[t]) / peak);
        }
        int nTrades = trades.size();
        int wins = 0;
        long holdSum = 0;
        for (Trade t : trades) {
            if (t.pnl() > 0.0) {
                wins++;
            }
            holdSum += t.holdingDays();
        }
        Map<String, Double> out = new LinkedHashMap<>();
        out.put("total_return", totalReturn);
        out.put("ann_return", annReturn);
        out.put("sharpe", sharpe);
        out.put("max_drawdown", maxDd);
        out.put("n_trades", (double) nTrades);
        out.put("win_rate", nTrades > 0 ? (double) wins / nTrades : 0.0);
        out.put("avg_holding_days", nTrades > 0 ? (double) holdSum / nTrades : 0.0);
        return out;
    }

    /**
     * Run the pinned pairs state machine over aligned price series.
     *
     * @param y      close price series, strictly positive
     * @param x      close price series, equal length, strictly positive
     * @param alpha  hedge intercept defining the spread (estimated elsewhere)
     * @param beta   hedge ratio defining the spread
     * @param config validated signal/cost parameters
     * @throws IllegalArgumentException on length mismatch, non-finite or
     *     non-positive prices, beta &le; 0 or NaN, or window larger than the series
     */
    public static Result run(double[] y, double[] x, double alpha, double beta,
                             Config config) {
        return runFrom(y, x, alpha, beta, config, 0);
    }

    /** State machine; entry signals are ignored before {@code firstEntrySignal}. */
    private static Result runFrom(double[] y, double[] x, double alpha, double beta,
                                  Config config, int firstEntrySignal) {
        if (!(beta > 0.0)) {
            throw new IllegalArgumentException(
                    "beta must be > 0: negative hedges are out of scope");
        }
        if (y.length != x.length) {
            throw new IllegalArgumentException("y and x must be 1-D arrays of equal length");
        }
        if (!(LinAlg.allFinite(y) && LinAlg.allFinite(x))) {
            throw new IllegalArgumentException("prices must be finite (no NaN/inf)");
        }
        for (int t = 0; t < y.length; t++) {
            if (y[t] <= 0.0 || x[t] <= 0.0) {
                throw new IllegalArgumentException("prices must be strictly positive");
            }
        }

        double[] s = Signals.spread(y, x, alpha, beta);
        double[] z = Signals.rollingZscore(s, config.window());
        double bps = (config.costBps() + config.slippageBps()) / 1.0e4;
        double cap = config.capital();

        int n = y.length;
        double[] equity = new double[n];
        double[] dailyPnl = new double[n];
        double[] dailyCost = new double[n];
        List<Trade> trades = new ArrayList<>();

        int state = FLAT;
        double qy = 0.0;
        double qx = 0.0;
        // pending order: 0 = none, +1/-1 = enter long/short, 2 = exit, 3 = stop exit
        int pending = 0;
        int noEntryBefore = firstEntrySignal; // also moved by the cool-down
        int openEntryDay = -1;
        int openDir = 0;
        double openPnl = 0.0;
        double turnoverNotional = 0.0;
        double eq = cap;

        for (int t = 0; t < n; t++) {
            // 1) mark-to-market the position held since the day t-1 close
            double pnl = t > 0 ? qy * (y[t] - y[t - 1]) + qx * (x[t] - x[t - 1]) : 0.0;
            if (state != FLAT) {
                openPnl += pnl;
            }
            double cost = 0.0;

            // 2) execute the order decided at the day t-1 close.  A pending
            //    ENTRY on the final day is discarded (pinned): it could only be
            //    force-closed at the same close, a phantom zero-day round trip.
            if (pending != 0) {
                if ((pending == 1 || pending == -1) && state == FLAT && t < n - 1) {
                    int dir = pending;
                    if (BETA_HEDGED.equals(config.sizing())) {
                        double u = cap / (y[t] + beta * x[t]);
                        qy = dir * u;
                        qx = -dir * beta * u;
                    } else {
                        qy = dir * (cap / 2.0) / y[t];
                        qx = -dir * (cap / 2.0) / x[t];
                    }
                    double traded = Math.abs(qy) * y[t] + Math.abs(qx) * x[t];
                    cost += traded * bps;
                    turnoverNotional += traded;
                    state = dir;
                    openEntryDay = t;
                    openDir = dir;
                    openPnl = -traded * bps;
                } else if ((pending == 2 || pending == 3) && state != FLAT) {
                    double traded = Math.abs(qy) * y[t] + Math.abs(qx) * x[t];
                    cost += traded * bps;
                    turnoverNotional += traded;
                    openPnl -= traded * bps;
                    trades.add(new Trade(openEntryDay, t, openDir, openPnl, pending == 3));
                    if (pending == 3) {
                        noEntryBefore = Math.max(noEntryBefore, t + config.cooldownDays());
                    }
                    qy = 0.0;
                    qx = 0.0;
                    state = FLAT;
                }
                pending = 0;
            }

            // 2b) force-close a position still open at the final close
            if (t == n - 1 && state != FLAT) {
                double traded = Math.abs(qy) * y[t] + Math.abs(qx) * x[t];
                cost += traded * bps;
                turnoverNotional += traded;
                openPnl -= traded * bps;
                trades.add(new Trade(openEntryDay, t, openDir, openPnl, false));
                qy = 0.0;
                qx = 0.0;
                state = FLAT;
            }

            eq += pnl - cost;
            equity[t] = eq;
            dailyPnl[t] = pnl;
            dailyCost[t] = cost;

            // 3) generate the order to execute at the day t+1 close
            if (t < n - 1) {
                double zt = z[t];
                boolean zOk = !Double.isNaN(zt);
                if (state == FLAT) {
                    if (zOk && t >= noEntryBefore) {
                        if (config.entryZ() <= zt && zt < config.stopZ()) {
                            pending = -1;
                        } else if (-config.stopZ() < zt && zt <= -config.entryZ()) {
                            pending = 1;
                        }
                    }
                } else if (config.pnlStopFrac() > 0.0
                        && openPnl <= -config.pnlStopFrac() * cap) {
                    pending = 3;
                } else if (zOk && Math.abs(zt) >= config.stopZ()) {
                    pending = 3;
                } else if (config.maxHoldDays() > 0
                        && t - openEntryDay >= config.maxHoldDays()) {
                    pending = 2;
                } else if (zOk && Math.abs(zt) <= config.exitZ()) {
                    pending = 2;
                }
            }
        }

        Map<String, Double> metrics = computeMetrics(equity, cap, trades);
        metrics.put("turnover", turnoverNotional / cap);
        return new Result(equity, dailyPnl, dailyCost, z, trades, metrics);
    }

    /**
     * Estimate the hedge on the first {@code insampleFrac} of days by OLS,
     * backtest that segment in-sample, then trade the remainder out-of-sample
     * with the SAME (alpha, beta) — degradation on display.
     *
     * <p>This is a SINGLE HOLDOUT (one fit, one split); the name is kept for
     * compatibility, see {@link #rollingWalkForward} for the refitting version.
     *
     * <p>{@code split = floor(T * insampleFrac + SPLIT_EPS)} (pinned). The
     * out-of-sample z-score warms up inside the OOS segment only (no
     * information leaks across the split).
     *
     * @throws IllegalArgumentException when insampleFrac is outside (0, 1), the
     *     series differ in length, or either segment is shorter than the
     *     z-score window
     */
    public static WalkForwardResult walkForward(double[] y, double[] x, Config config,
                                                double insampleFrac) {
        if (!(insampleFrac > 0.0 && insampleFrac < 1.0)) {
            throw new IllegalArgumentException("insample_frac must lie strictly inside (0, 1)");
        }
        if (y.length != x.length) {
            throw new IllegalArgumentException("y and x must be 1-D arrays of equal length");
        }
        int split = (int) Math.floor(y.length * insampleFrac + SPLIT_EPS);
        if (split < config.window() || y.length - split < config.window()) {
            throw new IllegalArgumentException("each walk-forward segment must cover the z window");
        }
        double[] yIn = Arrays.copyOfRange(y, 0, split);
        double[] xIn = Arrays.copyOfRange(x, 0, split);
        Ols.Result fit = Ols.fit(yIn, xIn, true);
        double a = fit.params()[0];
        double b = fit.params()[1];
        Result ins = run(yIn, xIn, a, b, config);
        Result oos = run(Arrays.copyOfRange(y, split, y.length),
                Arrays.copyOfRange(x, split, x.length), a, b, config);
        return new WalkForwardResult(a, b, ins, oos, split);
    }

    /**
     * One rolling test block: days [start, end) traded with a fit from before
     * start. {@code result} covers the block days only (trade days are absolute).
     *
     * @param start   first block day
     * @param end     one past the last block day
     * @param alpha   fitted intercept
     * @param beta    fitted hedge ratio
     * @param skipped true when the fit gave beta &le; 0 or NaN and the block stays flat
     * @param result  block backtest trimmed to days [start, end)
     */
    public record RollingBlock(int start, int end, double alpha, double beta,
                               boolean skipped, Result result) {
    }

    /**
     * Stitched rolling walk-forward; {@code equity[i]} is day {@code startDay + i}.
     *
     * @param blocks   per-block results
     * @param startDay first stitched day (= train)
     * @param equity   stitched equity curve
     * @param trades   all blocks' trades
     * @param metrics  metrics of the stitched curve (turnover = sum of blocks)
     */
    public record RollingResult(List<RollingBlock> blocks, int startDay, double[] equity,
                                List<Trade> trades, Map<String, Double> metrics) {

        /** Number of stopped trades across all blocks. */
        public int nStops() {
            int c = 0;
            for (Trade t : trades) {
                if (t.stopped()) {
                    c++;
                }
            }
            return c;
        }
    }

    /**
     * Refit on a rolling window and trade only the days after each fit.
     *
     * <p>For k = 0, 1, .. while {@code k*test + train < T}: OLS on
     * {@code [k*test, k*test+train)}, trade {@code [start, end)} with
     * {@code start = k*test+train}, {@code end = min(start+test, T)}. Each block
     * is its own backtest on the W-1 prior days plus the block, entries ignored
     * before {@code start}, flat at block end (forced close, not a stop),
     * cool-down not carried. A block with beta &le; 0 or NaN stays flat and is
     * marked skipped.
     *
     * @throws IllegalArgumentException on length mismatch, test &lt; 1,
     *     train &lt; window, or train &ge; T
     */
    public static RollingResult rollingWalkForward(double[] y, double[] x, Config config,
                                                   int train, int test) {
        if (y.length != x.length) {
            throw new IllegalArgumentException("y and x must be 1-D arrays of equal length");
        }
        if (test < 1) {
            throw new IllegalArgumentException("test must be >= 1");
        }
        if (train < config.window()) {
            throw new IllegalArgumentException("train must cover the z window");
        }
        int n = y.length;
        if (train >= n) {
            throw new IllegalArgumentException("train must be shorter than the series");
        }
        double cap = config.capital();
        int warm = config.window() - 1;
        List<RollingBlock> blocks = new ArrayList<>();
        List<Trade> trades = new ArrayList<>();
        double[] stitched = new double[n - train];
        int filled = 0;
        double running = cap;
        double turnover = 0.0;
        for (long k = 0; k * test + train < n; k++) {
            int fitLo = (int) (k * test);
            int start = fitLo + train;
            int end = (int) Math.min((long) start + test, n);
            Ols.Result fit = Ols.fit(Arrays.copyOfRange(y, fitLo, start),
                    Arrays.copyOfRange(x, fitLo, start), true);
            double a = fit.params()[0];
            double b = fit.params()[1];
            boolean skipped = !(b > 0.0);
            Result res;
            if (skipped) {
                res = flatBlock(end - start, cap);
            } else {
                int lo = start - warm;
                Result full = runFrom(Arrays.copyOfRange(y, lo, end),
                        Arrays.copyOfRange(x, lo, end), a, b, config, warm);
                List<Trade> bt = new ArrayList<>();
                for (Trade t : full.trades()) {
                    bt.add(new Trade(t.entryDay() + lo, t.exitDay() + lo, t.direction(),
                            t.pnl(), t.stopped()));
                }
                double[] eq = Arrays.copyOfRange(full.equity(), warm, full.equity().length);
                Map<String, Double> m = computeMetrics(eq, cap, bt);
                m.put("turnover", full.metrics().get("turnover"));
                res = new Result(eq,
                        Arrays.copyOfRange(full.dailyPnl(), warm, full.dailyPnl().length),
                        Arrays.copyOfRange(full.dailyCost(), warm, full.dailyCost().length),
                        Arrays.copyOfRange(full.zscore(), warm, full.zscore().length),
                        bt, m);
            }
            blocks.add(new RollingBlock(start, end, a, b, skipped, res));
            trades.addAll(res.trades());
            for (double e : res.equity()) {
                stitched[filled++] = running + (e - cap);
            }
            running = stitched[filled - 1];
            turnover += res.metrics().get("turnover");
        }
        Map<String, Double> metrics = computeMetrics(stitched, cap, trades);
        metrics.put("turnover", turnover);
        return new RollingResult(blocks, train, stitched, trades, metrics);
    }

    private static Result flatBlock(int n, double cap) {
        double[] eq = new double[n];
        Arrays.fill(eq, cap);
        double[] z = new double[n];
        Arrays.fill(z, Double.NaN);
        List<Trade> none = new ArrayList<>();
        Map<String, Double> m = computeMetrics(eq, cap, none);
        m.put("turnover", 0.0);
        return new Result(eq, new double[n], new double[n], z, none, m);
    }
}
