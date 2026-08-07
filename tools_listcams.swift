// Print video capture devices in AVFoundation's own order — the same order
// OpenCV assigns indices. OpenCV exposes only indices, and macOS's
// system_profiler lists cameras in a different order, so this is the only
// reliable way to know which index is which camera.
import AVFoundation
import Foundation

// AVCaptureDevice.devices(for:) returns the system's own order — the same one
// OpenCV's AVFoundation backend assigns indices from. A DiscoverySession does
// NOT: it groups results by the deviceTypes you ask for, so builtin cameras
// come back before external ones and the mapping silently shifts.
let devices = AVCaptureDevice.devices(for: .video)

for (i, d) in devices.enumerated() {
    var kind = "other"
    if #available(macOS 14.0, *) {
        if d.deviceType == .continuityCamera { kind = "continuity" }
        else if d.deviceType == .external { kind = "external" }
        else if d.deviceType == .builtInWideAngleCamera { kind = "builtin" }
    }
    // tab-separated: index, kind, name
    print("\(i)\t\(kind)\t\(d.localizedName)")
}
