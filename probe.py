import cv2, time
for i in range(4):
    cap = cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        print("index %d: not opened" % i); cap.release(); continue
    ok, f = cap.read()
    if not ok or f is None:
        print("index %d: opened but no frames" % i); cap.release(); continue
    default = (f.shape[1], f.shape[0])
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 4656); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 3496)
    time.sleep(0.5)
    best = default
    t0 = time.time()
    while time.time() - t0 < 4:
        ok, f = cap.read()
        if ok and f is not None:
            best = (f.shape[1], f.shape[0])
            if best[0] >= 4000: break
    print("index %d: default %s  max-attempt %s" % (i, default, best))
    cap.release()
