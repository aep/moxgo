//go:build darwin

package camera

import (
	"fmt"
	"image"
	"runtime"
	"sync"
	"unsafe"

	"github.com/ebitengine/purego"
)

var (
	initOnce sync.Once
	initErr  error

	// ObjC runtime
	fnMsgSend              uintptr
	fnGetClass             uintptr
	fnSelRegisterName      uintptr
	fnAllocateClassPair    uintptr
	fnRegisterClassPair    uintptr
	fnClassAddMethod       uintptr
	fnClassAddIvar         uintptr
	fnClassGetIvar         uintptr
	fnObjectSetIvar        uintptr
	fnObjectGetIvar        uintptr

	// CoreMedia / CoreVideo
	fnCMSampleBufferGetImageBuffer uintptr
	fnCVPixelBufferLockBaseAddress uintptr
	fnCVPixelBufferUnlockBase      uintptr
	fnCVPixelBufferGetBaseAddress  uintptr
	fnCVPixelBufferGetWidth        uintptr
	fnCVPixelBufferGetHeight       uintptr
	fnCVPixelBufferGetBytesPerRow  uintptr

	// libdispatch
	fnDispatchQueueCreate uintptr

	// Global NSString constant: AVMediaTypeVideo
	avMediaTypeVideo uintptr

	// kCVPixelBufferPixelFormatTypeKey
	cvPixelBufferPixelFormatTypeKey uintptr
)

func ensureInit() error {
	initOnce.Do(func() { initErr = loadFrameworks() })
	return initErr
}

func loadFrameworks() error {
	libobjc, err := purego.Dlopen("/usr/lib/libobjc.A.dylib", purego.RTLD_LAZY)
	if err != nil {
		return fmt.Errorf("camera: load libobjc: %w", err)
	}
	avf, err := purego.Dlopen("/System/Library/Frameworks/AVFoundation.framework/AVFoundation", purego.RTLD_LAZY)
	if err != nil {
		return fmt.Errorf("camera: load AVFoundation: %w", err)
	}
	cm, err := purego.Dlopen("/System/Library/Frameworks/CoreMedia.framework/CoreMedia", purego.RTLD_LAZY)
	if err != nil {
		return fmt.Errorf("camera: load CoreMedia: %w", err)
	}
	cv, err := purego.Dlopen("/System/Library/Frameworks/CoreVideo.framework/CoreVideo", purego.RTLD_LAZY)
	if err != nil {
		return fmt.Errorf("camera: load CoreVideo: %w", err)
	}
	libsys, err := purego.Dlopen("/usr/lib/libSystem.B.dylib", purego.RTLD_LAZY)
	if err != nil {
		return fmt.Errorf("camera: load libSystem: %w", err)
	}

	fnMsgSend, _ = purego.Dlsym(libobjc, "objc_msgSend")
	fnGetClass, _ = purego.Dlsym(libobjc, "objc_getClass")
	fnSelRegisterName, _ = purego.Dlsym(libobjc, "sel_registerName")
	fnAllocateClassPair, _ = purego.Dlsym(libobjc, "objc_allocateClassPair")
	fnRegisterClassPair, _ = purego.Dlsym(libobjc, "objc_registerClassPair")
	fnClassAddMethod, _ = purego.Dlsym(libobjc, "class_addMethod")
	fnClassAddIvar, _ = purego.Dlsym(libobjc, "class_addIvar")
	fnClassGetIvar, _ = purego.Dlsym(libobjc, "class_getInstanceVariable")
	fnObjectSetIvar, _ = purego.Dlsym(libobjc, "object_setIvar")
	fnObjectGetIvar, _ = purego.Dlsym(libobjc, "object_getIvar")

	fnCMSampleBufferGetImageBuffer, _ = purego.Dlsym(cm, "CMSampleBufferGetImageBuffer")
	fnCVPixelBufferLockBaseAddress, _ = purego.Dlsym(cv, "CVPixelBufferLockBaseAddress")
	fnCVPixelBufferUnlockBase, _ = purego.Dlsym(cv, "CVPixelBufferUnlockBaseAddress")
	fnCVPixelBufferGetBaseAddress, _ = purego.Dlsym(cv, "CVPixelBufferGetBaseAddress")
	fnCVPixelBufferGetWidth, _ = purego.Dlsym(cv, "CVPixelBufferGetWidth")
	fnCVPixelBufferGetHeight, _ = purego.Dlsym(cv, "CVPixelBufferGetHeight")
	fnCVPixelBufferGetBytesPerRow, _ = purego.Dlsym(cv, "CVPixelBufferGetBytesPerRow")
	fnDispatchQueueCreate, _ = purego.Dlsym(libsys, "dispatch_queue_create")

	// Resolve global NSString constants
	if sym, err := purego.Dlsym(avf, "AVMediaTypeVideo"); err == nil {
		avMediaTypeVideo = *(*uintptr)(unsafe.Pointer(sym))
	}
	if sym, err := purego.Dlsym(cv, "kCVPixelBufferPixelFormatTypeKey"); err == nil {
		cvPixelBufferPixelFormatTypeKey = *(*uintptr)(unsafe.Pointer(sym))
	}

	return nil
}

// ObjC runtime helpers

func sel(name string) uintptr {
	b := append([]byte(name), 0)
	r, _, _ := purego.SyscallN(fnSelRegisterName, uintptr(unsafe.Pointer(&b[0])))
	runtime.KeepAlive(b)
	return r
}

func cls(name string) uintptr {
	b := append([]byte(name), 0)
	r, _, _ := purego.SyscallN(fnGetClass, uintptr(unsafe.Pointer(&b[0])))
	runtime.KeepAlive(b)
	return r
}

func msg(obj, s uintptr, args ...uintptr) uintptr {
	all := make([]uintptr, 0, 2+len(args))
	all = append(all, obj, s)
	all = append(all, args...)
	r, _, _ := purego.SyscallN(fnMsgSend, all...)
	return r
}

func nsStringToGo(ns uintptr) string {
	if ns == 0 {
		return ""
	}
	cstr := msg(ns, sel("UTF8String"))
	if cstr == 0 {
		return ""
	}
	var n int
	for *(*byte)(unsafe.Pointer(cstr + uintptr(n))) != 0 {
		n++
	}
	return string(unsafe.Slice((*byte)(unsafe.Pointer(cstr)), n))
}

func goToNSString(s string) uintptr {
	b := append([]byte(s), 0)
	r := msg(cls("NSString"), sel("stringWithUTF8String:"), uintptr(unsafe.Pointer(&b[0])))
	runtime.KeepAlive(b)
	return r
}

// Camera registry — maps delegate ivar value to Go camera struct.

var (
	camMu    sync.RWMutex
	cameras  = map[uintptr]*avCamera{}
	nextID   uintptr
)

func registerCam(c *avCamera) uintptr {
	camMu.Lock()
	defer camMu.Unlock()
	nextID++
	cameras[nextID] = c
	return nextID
}

func unregisterCam(id uintptr) {
	camMu.Lock()
	defer camMu.Unlock()
	delete(cameras, id)
}

// Delegate class — created once, reused for all cameras.

var (
	delegateOnce  sync.Once
	delegateClass uintptr
	delegateIvar  uintptr // Ivar handle for "camID"
)

func ensureDelegate() {
	delegateOnce.Do(func() {
		nsObject := cls("NSObject")
		name := append([]byte("MoxgoCameraDelegate"), 0)
		delegateClass, _, _ = purego.SyscallN(fnAllocateClassPair,
			nsObject, uintptr(unsafe.Pointer(&name[0])), 0)
		runtime.KeepAlive(name)

		ivarName := append([]byte("camID"), 0)
		ivarType := append([]byte("Q"), 0)
		purego.SyscallN(fnClassAddIvar, delegateClass,
			uintptr(unsafe.Pointer(&ivarName[0])),
			8, 3, // size=8, log2(alignment)=3
			uintptr(unsafe.Pointer(&ivarType[0])))
		runtime.KeepAlive(ivarName)
		runtime.KeepAlive(ivarType)

		methodSel := sel("captureOutput:didOutputSampleBuffer:fromConnection:")
		methodTypes := append([]byte("v@:@@@"), 0)
		purego.SyscallN(fnClassAddMethod, delegateClass, methodSel,
			frameCallbackPtr, uintptr(unsafe.Pointer(&methodTypes[0])))
		runtime.KeepAlive(methodTypes)

		purego.SyscallN(fnRegisterClassPair, delegateClass)

		ivarNameB := append([]byte("camID"), 0)
		delegateIvar, _, _ = purego.SyscallN(fnClassGetIvar,
			delegateClass, uintptr(unsafe.Pointer(&ivarNameB[0])))
		runtime.KeepAlive(ivarNameB)
	})
}

var frameCallbackPtr = purego.NewCallback(func(self, _cmd, output, sampleBuffer, connection uintptr) {
	camID, _, _ := purego.SyscallN(fnObjectGetIvar, self, delegateIvar)

	camMu.RLock()
	cam, ok := cameras[camID]
	camMu.RUnlock()
	if !ok {
		return
	}

	imageBuffer, _, _ := purego.SyscallN(fnCMSampleBufferGetImageBuffer, sampleBuffer)
	if imageBuffer == 0 {
		return
	}
	purego.SyscallN(fnCVPixelBufferLockBaseAddress, imageBuffer, 0)

	baseAddr, _, _ := purego.SyscallN(fnCVPixelBufferGetBaseAddress, imageBuffer)
	width, _, _ := purego.SyscallN(fnCVPixelBufferGetWidth, imageBuffer)
	height, _, _ := purego.SyscallN(fnCVPixelBufferGetHeight, imageBuffer)
	bpr, _, _ := purego.SyscallN(fnCVPixelBufferGetBytesPerRow, imageBuffer)

	if baseAddr == 0 || width == 0 || height == 0 {
		purego.SyscallN(fnCVPixelBufferUnlockBase, imageBuffer, 0)
		return
	}

	w, h := int(width), int(height)
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	src := unsafe.Slice((*byte)(unsafe.Pointer(baseAddr)), int(height)*int(bpr))
	for y := range h {
		si := y * int(bpr)
		di := y * img.Stride
		for range w {
			img.Pix[di+0] = src[si+2] // BGRA → RGBA
			img.Pix[di+1] = src[si+1]
			img.Pix[di+2] = src[si+0]
			img.Pix[di+3] = src[si+3]
			si += 4
			di += 4
		}
	}

	purego.SyscallN(fnCVPixelBufferUnlockBase, imageBuffer, 0)

	select {
	case <-cam.frames:
	default:
	}
	cam.frames <- img
})

// avCamera wraps an AVCaptureSession.

type avCamera struct {
	session  uintptr
	delegate uintptr
	camID    uintptr
	frames   chan image.Image
	done     chan struct{}
	closed   bool
}

func list() ([]Device, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}

	avCaptureDevice := cls("AVCaptureDevice")
	devices := msg(avCaptureDevice, sel("devicesWithMediaType:"), avMediaTypeVideo)
	count := msg(devices, sel("count"))
	var result []Device
	for i := uintptr(0); i < count; i++ {
		dev := msg(devices, sel("objectAtIndex:"), i)
		name := nsStringToGo(msg(dev, sel("localizedName")))
		uid := nsStringToGo(msg(dev, sel("uniqueID")))
		result = append(result, Device{Index: int(i), Name: name, Path: uid})
	}
	return result, nil
}

// TODO: honor opts.Width/Height (AVCaptureSession preset or video settings).
func open(index int, _ Options) (Camera, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	ensureDelegate()

	devs, err := list()
	if err != nil {
		return nil, err
	}
	if index >= len(devs) {
		return nil, fmt.Errorf("camera: index %d out of range (%d devices found)", index, len(devs))
	}

	uid := goToNSString(devs[index].Path)
	device := msg(cls("AVCaptureDevice"), sel("deviceWithUniqueID:"), uid)
	if device == 0 {
		return nil, fmt.Errorf("camera: device not found")
	}

	var errPtr uintptr
	input := msg(cls("AVCaptureDeviceInput"), sel("deviceInputWithDevice:error:"),
		device, uintptr(unsafe.Pointer(&errPtr)))
	if input == 0 {
		return nil, fmt.Errorf("camera: failed to create capture input (check camera permissions)")
	}

	session := msg(msg(cls("AVCaptureSession"), sel("alloc")), sel("init"))
	msg(session, sel("addInput:"), input)

	output := msg(msg(cls("AVCaptureVideoDataOutput"), sel("alloc")), sel("init"))

	// Set pixel format to BGRA
	const kCVPixelFormatType32BGRA = 0x42475241
	nsNum := msg(cls("NSNumber"), sel("numberWithUnsignedInt:"), uintptr(kCVPixelFormatType32BGRA))
	settings := msg(cls("NSDictionary"), sel("dictionaryWithObject:forKey:"),
		nsNum, cvPixelBufferPixelFormatTypeKey)
	msg(output, sel("setVideoSettings:"), settings)

	cam := &avCamera{
		session: session,
		frames:  make(chan image.Image, 1),
		done:    make(chan struct{}),
	}
	cam.camID = registerCam(cam)

	delegate := msg(msg(delegateClass, sel("alloc")), sel("init"))
	purego.SyscallN(fnObjectSetIvar, delegate, delegateIvar, cam.camID)
	cam.delegate = delegate

	queueLabel := append([]byte("moxgo.camera"), 0)
	queue, _, _ := purego.SyscallN(fnDispatchQueueCreate,
		uintptr(unsafe.Pointer(&queueLabel[0])), 0)
	runtime.KeepAlive(queueLabel)

	msg(output, sel("setSampleBufferDelegate:queue:"), delegate, queue)
	msg(session, sel("addOutput:"), output)
	msg(session, sel("startRunning"))

	return cam, nil
}

func (c *avCamera) NextFrame() (image.Image, error) {
	select {
	case img := <-c.frames:
		return img, nil
	case <-c.done:
		return nil, fmt.Errorf("camera: closed")
	}
}

func (c *avCamera) Close() error {
	if c.closed {
		return nil
	}
	c.closed = true
	msg(c.session, sel("stopRunning"))
	unregisterCam(c.camID)
	close(c.done)
	msg(c.delegate, sel("release"))
	msg(c.session, sel("release"))
	return nil
}
