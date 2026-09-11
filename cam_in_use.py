"""Which video devices are currently in use by any process (CoreMediaIO DeviceIsRunningSomewhere)."""
import struct, objc, CoreMediaIO as CM
from Foundation import NSString

def _cc(s): return struct.unpack(">I", s.encode())[0]
SYSTEM = 1  # kCMIOObjectSystemObject

def _get(obj, sel):
    a = CM.CMIOObjectPropertyAddress(_cc(sel), _cc("glob"), 0)
    err, n = CM.CMIOObjectGetPropertyDataSize(obj, a, 0, None, None)
    if err: raise OSError(err)
    r = CM.CMIOObjectGetPropertyData(obj, a, 0, None, n, None, None)
    if r[0]: raise OSError(r[0])
    return bytes(r[-1])[: r[-2]]

def _name(did):
    ptr = struct.unpack("<Q", _get(did, "lnam"))[0]   # kCMIOObjectPropertyName -> CFStringRef
    return str(objc.objc_object(c_void_p=ptr))

def devices():
    """{device_id: name} for every CoreMediaIO video device."""
    return {did: _name(did) for (did,) in struct.iter_unpack("<I", _get(SYSTEM, "dev#"))}

def running(did):
    # kCMIODevicePropertyDeviceIsRunningSomewhere
    return struct.unpack("<I", _get(did, "gone")[:4])[0] != 0

def in_use(name_substr):
    """True if any process is streaming from a device whose name contains name_substr."""
    return any(running(d) for d, n in devices().items() if name_substr.lower() in n.lower())

if __name__ == "__main__":
    for did, name in devices().items():
        try: print(f"{did:4} {name:28} running={running(did)}")
        except OSError as e: print(did, name, "err", e)
