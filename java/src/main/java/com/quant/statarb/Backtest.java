package com.quant.statarb;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Dollar-neutral pairs backtest — state machine and accounting (PINNED).
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
 *   <li>FLAT, signal day t: {@code entry_z <= z_t < stop_z} -&gt; pending SHORT
 *       entry; {@code -stop_z < z_t <= -entry_z} -&gt; pending LONG entry
 *       ({@code |z_t| >= stop_z} while flat: no entry, pinned; NaN z: no
 *       action).</li>
 *   <li>In position, signal day t: {@code |z_t| >= stop_z} -&gt; pending STOP
 *       exit; else {@code |z_t| <= exit_z} -&gt; pending exit.</li>
 *   <li>Execution day t+1 close. Entry legs are EQUAL-DOLLAR (pinned):
 *       {@code q_y = dir * (capital/2) / y}, {@code q_x = -dir * (capital/2) / x}
 *       with dir = +1 for LONG spread (long y / short x), -1 for SHORT. The
 *       hedge ratio enters through the spread/signal; sizing is equal-dollar so
 *       the position is exactly dollar-neutral at entry. Quantities are then
 *       held fixed (no daily rebalance).</li>
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

    /**
     * Signal / cost parameters for the pairs backtest, validated on
     * construction.
     *
     * @param window      rolling z-score window W (&ge; 2)
     * @param entryZ      enter when |z| &ge; entryZ (and &lt; stopZ)
     * @param exitZ       exit when |z| &le; exitZ
     * @param stopZ       stop-loss when |z| &ge; stopZ
     * @param costBps     commission, basis points of traded notional
     * @param slippageBps slippage, basis points of traded notional
     * @param capital     gross capital; each leg is sized to capital/2 at entry
     */
    public record Config(int window, double entryZ, double exitZ, double stopZ,
                         double costBps, double slippageBps, double capital) {

        /**
         * Validates the pinned parameter constraints.
         *
         * @throws IllegalArgumentException when window &lt; 2, any threshold or
         *     cost is NaN/infinite, the ordering is violated, costs are negative,
         *     or capital is not positive and finite
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
     *     non-positive prices, or window larger than the series
     */
    public static Result run(double[] y, double[] x, double alpha, double beta,
                             Config config) {
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
                    qy = dir * (cap / 2.0) / y[t];
                    qx = -dir * (cap / 2.0) / x[t];
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
                if (!Double.isNaN(zt)) {
                    if (state == FLAT) {
                        if (config.entryZ() <= zt && zt < config.stopZ()) {
                            pending = -1;
                        } else if (-config.stopZ() < zt && zt <= -config.entryZ()) {
                            pending = 1;
                        }
                    } else {
                        if (Math.abs(zt) >= config.stopZ()) {
                            pending = 3;
                        } else if (Math.abs(zt) <= config.exitZ()) {
                            pending = 2;
                        }
                    }
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
        double[] yIn = java.util.Arrays.copyOfRange(y, 0, split);
        double[] xIn = java.util.Arrays.copyOfRange(x, 0, split);
        Ols.Result fit = Ols.fit(yIn, xIn, true);
        double a = fit.params()[0];
        double b = fit.params()[1];
        Result ins = run(yIn, xIn, a, b, config);
        Result oos = run(java.util.Arrays.copyOfRange(y, split, y.length),
                java.util.Arrays.copyOfRange(x, split, x.length), a, b, config);
        return new WalkForwardResult(a, b, ins, oos, split);
    }
}
