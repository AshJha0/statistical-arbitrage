#!/usr/bin/env bash
# Build the statarb library, demo and tests into out/ with plain javac (Maven users: `mvn test`).
# Test compilation needs JUnit 4 + Hamcrest jars: JUNIT_JAR / HAMCREST_JAR, else found under
# $JUNIT_HOME, /usr/share/java or ~/.m2.
set -euo pipefail
cd "$(dirname "$0")"
# classpath separator: ';' under Git Bash / MSYS / Cygwin (native Windows java), ':' elsewhere
case "$(uname -s 2>/dev/null)" in
  MINGW*|MSYS*|CYGWIN*) SEP=';' ;;
  *) SEP=':' ;;
esac

find_jar() { # name-pattern
  { [ -n "${JUNIT_HOME:-}" ] && find "$JUNIT_HOME" -name "$1" 2>/dev/null
    find /usr/share/java "$HOME/.m2" -name "$1" 2>/dev/null; } | head -n1 || true
}
JUNIT="${JUNIT_JAR:-$(find_jar 'junit*.jar')}"
HAMCREST="${HAMCREST_JAR:-$(find_jar 'hamcrest*.jar')}"

rm -rf out
mkdir -p out/main out/test

javac -Xlint:all -Werror --release 21 -d out/main $(find src/main/java -name '*.java')
if [ -z "$JUNIT" ] || [ -z "$HAMCREST" ]; then
  echo "build OK (main only: JUnit/Hamcrest jars not found, tests not compiled)"
  exit 0
fi
javac -Xlint:all -Werror --release 21 -cp "out/main${SEP}$JUNIT${SEP}$HAMCREST" -d out/test \
    $(find src/test/java -name '*.java')

echo "build OK"
