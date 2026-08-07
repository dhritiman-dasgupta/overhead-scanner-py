#!/bin/bash
# Build "Overhead Scanner.app" — a wrapper around scanner.py, not a copy of it.
#
#     ./make-app.sh && open "Overhead Scanner.app"
#
# Worth the twenty lines. Run as a bare script the app is called "Python" in
# the menu bar, has the generic rocket icon in the Dock, and macOS attributes
# the camera permission to the interpreter rather than to this app. A bundle
# fixes all three, and because MacOS/launcher only execs the checked-out source
# there is nothing to rebuild after an edit.
set -e
cd "$(dirname "$0")"
SRC="$(pwd -P)"
APP="$SRC/Overhead Scanner.app"

rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"

cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key>              <string>Overhead Scanner</string>
  <key>CFBundleDisplayName</key>       <string>Overhead Scanner</string>
  <key>CFBundleIdentifier</key>        <string>local.overhead-scanner</string>
  <key>CFBundleVersion</key>           <string>1.0</string>
  <key>CFBundleShortVersionString</key><string>1.0</string>
  <key>CFBundlePackageType</key>       <string>APPL</string>
  <key>CFBundleExecutable</key>        <string>launcher</string>
  <key>CFBundleIconFile</key>          <string>icon</string>
  <key>NSHighResolutionCapable</key>   <true/>
  <key>LSApplicationCategoryType</key> <string>public.app-category.productivity</string>
  <!-- Without this macOS kills the app the moment it opens a camera. -->
  <key>NSCameraUsageDescription</key>
  <string>Overhead Scanner reads pages through your document camera.</string>
</dict>
</plist>
PLIST

# A compiled executable, not a shell script. macOS 26 will not launch a bundle
# whose CFBundleExecutable is a script: `open` returns success, nothing starts,
# and nothing is written to any log. Running the same script by hand works,
# which makes it a thoroughly misleading failure. Five lines of Swift avoids it.
cat > /tmp/ohs-launcher.swift <<SWIFT
import Darwin
import Foundation
FileManager.default.changeCurrentDirectoryPath("$SRC")
let py = "/usr/bin/python3", script = "$SRC/scanner.py"
var argv: [UnsafeMutablePointer<CChar>?] = [strdup(py), strdup(script), nil]
execv(py, &argv)
FileHandle.standardError.write("could not exec \(py)\n".data(using: .utf8)!)
exit(1)
SWIFT
if command -v swiftc >/dev/null; then
  swiftc -O /tmp/ohs-launcher.swift -o "$APP/Contents/MacOS/launcher"
else
  printf '#!/bin/bash\ncd "%s"\nexec /usr/bin/python3 "%s/scanner.py" "$@"\n' "$SRC" "$SRC" \
    > "$APP/Contents/MacOS/launcher"
  echo "  (no swiftc: falling back to a script — 'open' may not work, run scanner.py directly)"
fi
chmod +x "$APP/Contents/MacOS/launcher"

python3 make-icon.py "$APP/Contents/Resources" || echo "  (no icon — carrying on)"

# Ad-hoc signature. Without one macOS 26 refuses to launch the bundle at all:
# `open` returns success and nothing happens, while running the executable
# directly works fine — which is a confusing way to spend an afternoon.
codesign --force --deep --sign - "$APP" 2>/dev/null && echo "  signed (ad-hoc)"

echo "built: $APP"
echo "run:   open \"$APP\""
