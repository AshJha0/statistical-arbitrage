#!/usr/bin/env bash
# Run the JUnit tests; any cwd. Uses Maven when installed (no build.sh needed),
# otherwise falls back to JUnitCore on the classes from `bash build.sh`.
# Fallback jars: JUNIT_JAR / HAMCREST_JAR, else found under $JUNIT_HOME, /usr/share/java or ~/.m2.
set -euo pipefail
cd "$(dirname "$0")"
# classpath separator: ';' under Git Bash / MSYS / Cygwin (native Windows java), ':' elsewhere
case "$(uname -s 2>/dev/null)" in
  MINGW*|MSYS*|CYGWIN*) SEP=';' ;;
  *) SEP=':' ;;
esac

if command -v mvn >/dev/null 2>&1; then
  exec mvn -B -q test "$@"
fi

find_jar() { # name-pattern
  { [ -n "${JUNIT_HOME:-}" ] && find "$JUNIT_HOME" -name "$1" 2>/dev/null
    find /usr/share/java "$HOME/.m2" -name "$1" 2>/dev/null; } | head -n1 || true
}
JUNIT="${JUNIT_JAR:-$(find_jar 'junit*.jar')}"
HAMCREST="${HAMCREST_JAR:-$(find_jar 'hamcrest*.jar')}"
if [ -z "$JUNIT" ] || [ -z "$HAMCREST" ]; then
  echo "Maven not found and no JUnit/Hamcrest jars located; install Maven or set JUNIT_JAR/HAMCREST_JAR" >&2
  exit 1
fi

TESTS=$(find out/test -name '*Test.class' \
    | sed -e 's|^out/test/||' -e 's|\.class$||' -e 's|/|.|g' | sort)

java -cp "out/main${SEP}out/test${SEP}$JUNIT${SEP}$HAMCREST" org.junit.runner.JUnitCore $TESTS
