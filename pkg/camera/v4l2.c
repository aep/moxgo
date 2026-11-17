//go:build cgo

#include <linux/videodev2.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <poll.h>
#include <string.h>
#include <errno.h>

// All functions taking a buf_type accept either V4L2_BUF_TYPE_VIDEO_CAPTURE
// (single-planar API) or V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE (multi-planar API).
// Per-plane arrays passed in from Go must hold at least VIDEO_MAX_PLANES entries.

static int is_mplane(unsigned int type) {
    return type == V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
}

static int xioctl(int fd, unsigned long req, void *arg) {
    int r;
    do {
        r = ioctl(fd, req, arg);
    } while (r < 0 && errno == EINTR);
    return r;
}

// v4l2_open opens a video device. Returns the fd, or -errno on failure.
int v4l2_open(const char *path) {
    int fd = open(path, O_RDWR | O_NONBLOCK);
    if (fd < 0) return -errno;
    return fd;
}

// v4l2_close closes a file descriptor.
int v4l2_close(int fd) {
    return close(fd);
}

// v4l2_querycap fills name (up to name_len) and returns the capabilities of
// this specific device node. A physical device can expose several nodes
// (e.g. UVC capture + metadata); cap.capabilities describes all of them, so
// device_caps is used when the driver provides it.
// Returns 0 on success, -errno on failure.
int v4l2_querycap(int fd, char *name, int name_len, unsigned int *caps) {
    struct v4l2_capability cap;
    memset(&cap, 0, sizeof(cap));
    if (xioctl(fd, VIDIOC_QUERYCAP, &cap) < 0)
        return -errno;
    int n = strlen((char *)cap.card);
    if (n >= name_len) n = name_len - 1;
    memcpy(name, cap.card, n);
    name[n] = 0;
    if (cap.capabilities & V4L2_CAP_DEVICE_CAPS)
        *caps = cap.device_caps;
    else
        *caps = cap.capabilities;
    return 0;
}

// v4l2_enum_fmt returns the pixel format at index. Returns -EINVAL past the end.
int v4l2_enum_fmt(int fd, unsigned int type, unsigned int index, unsigned int *pixfmt) {
    struct v4l2_fmtdesc d;
    memset(&d, 0, sizeof(d));
    d.type = type;
    d.index = index;
    if (xioctl(fd, VIDIOC_ENUM_FMT, &d) < 0)
        return -errno;
    *pixfmt = d.pixelformat;
    return 0;
}

// v4l2_set_format sets the capture format. The driver may adjust any value;
// the actual format, size, plane count and per-plane bytesperline are returned.
// Returns 0 on success, -errno on failure.
int v4l2_set_format(int fd, unsigned int type, unsigned int pixfmt, unsigned int w, unsigned int h,
                    unsigned int *actual_pixfmt, unsigned int *actual_w, unsigned int *actual_h,
                    unsigned int *num_planes, unsigned int *bytesperline) {
    struct v4l2_format fmt;
    memset(&fmt, 0, sizeof(fmt));
    fmt.type = type;
    if (is_mplane(type)) {
        fmt.fmt.pix_mp.width = w;
        fmt.fmt.pix_mp.height = h;
        fmt.fmt.pix_mp.pixelformat = pixfmt;
        fmt.fmt.pix_mp.field = V4L2_FIELD_NONE;
    } else {
        fmt.fmt.pix.width = w;
        fmt.fmt.pix.height = h;
        fmt.fmt.pix.pixelformat = pixfmt;
        fmt.fmt.pix.field = V4L2_FIELD_NONE;
    }
    if (xioctl(fd, VIDIOC_S_FMT, &fmt) < 0)
        return -errno;
    if (is_mplane(type)) {
        *actual_pixfmt = fmt.fmt.pix_mp.pixelformat;
        *actual_w = fmt.fmt.pix_mp.width;
        *actual_h = fmt.fmt.pix_mp.height;
        unsigned int n = fmt.fmt.pix_mp.num_planes;
        if (n < 1) n = 1;
        if (n > VIDEO_MAX_PLANES) n = VIDEO_MAX_PLANES;
        *num_planes = n;
        for (unsigned int i = 0; i < n; i++)
            bytesperline[i] = fmt.fmt.pix_mp.plane_fmt[i].bytesperline;
    } else {
        *actual_pixfmt = fmt.fmt.pix.pixelformat;
        *actual_w = fmt.fmt.pix.width;
        *actual_h = fmt.fmt.pix.height;
        *num_planes = 1;
        bytesperline[0] = fmt.fmt.pix.bytesperline;
    }
    return 0;
}

// v4l2_request_buffers requests mmap buffers. count is in/out.
int v4l2_request_buffers(int fd, unsigned int type, unsigned int *count) {
    struct v4l2_requestbuffers req;
    memset(&req, 0, sizeof(req));
    req.count = *count;
    req.type = type;
    req.memory = V4L2_MEMORY_MMAP;
    if (xioctl(fd, VIDIOC_REQBUFS, &req) < 0)
        return -errno;
    *count = req.count;
    return 0;
}

// v4l2_query_buffer returns the mmap offset and length of each plane of a buffer.
int v4l2_query_buffer(int fd, unsigned int type, unsigned int index, unsigned int num_planes,
                      unsigned int *offsets, unsigned int *lengths) {
    struct v4l2_buffer buf;
    struct v4l2_plane planes[VIDEO_MAX_PLANES];
    memset(&buf, 0, sizeof(buf));
    memset(planes, 0, sizeof(planes));
    buf.type = type;
    buf.memory = V4L2_MEMORY_MMAP;
    buf.index = index;
    if (is_mplane(type)) {
        buf.m.planes = planes;
        buf.length = num_planes;
    }
    if (xioctl(fd, VIDIOC_QUERYBUF, &buf) < 0)
        return -errno;
    if (is_mplane(type)) {
        for (unsigned int i = 0; i < num_planes; i++) {
            offsets[i] = planes[i].m.mem_offset;
            lengths[i] = planes[i].length;
        }
    } else {
        offsets[0] = buf.m.offset;
        lengths[0] = buf.length;
    }
    return 0;
}

// v4l2_mmap maps a buffer plane.
void *v4l2_mmap(int fd, unsigned int offset, unsigned int length) {
    void *p = mmap(NULL, length, PROT_READ | PROT_WRITE, MAP_SHARED, fd, offset);
    if (p == MAP_FAILED) return NULL;
    return p;
}

// v4l2_munmap unmaps a buffer plane.
int v4l2_munmap(void *addr, unsigned int length) {
    return munmap(addr, length);
}

// v4l2_queue_buffer enqueues a buffer.
int v4l2_queue_buffer(int fd, unsigned int type, unsigned int index, unsigned int num_planes) {
    struct v4l2_buffer buf;
    struct v4l2_plane planes[VIDEO_MAX_PLANES];
    memset(&buf, 0, sizeof(buf));
    memset(planes, 0, sizeof(planes));
    buf.type = type;
    buf.memory = V4L2_MEMORY_MMAP;
    buf.index = index;
    if (is_mplane(type)) {
        buf.m.planes = planes;
        buf.length = num_planes;
    }
    if (xioctl(fd, VIDIOC_QBUF, &buf) < 0)
        return -errno;
    return 0;
}

// v4l2_dequeue_buffer dequeues a buffer. Returns the buffer index and, per
// plane, the bytes used and the offset at which valid data starts.
// Returns -EAGAIN if no buffer is ready (non-blocking).
int v4l2_dequeue_buffer(int fd, unsigned int type, unsigned int num_planes, unsigned int *index,
                        unsigned int *bytesused, unsigned int *data_offset) {
    struct v4l2_buffer buf;
    struct v4l2_plane planes[VIDEO_MAX_PLANES];
    memset(&buf, 0, sizeof(buf));
    memset(planes, 0, sizeof(planes));
    buf.type = type;
    buf.memory = V4L2_MEMORY_MMAP;
    if (is_mplane(type)) {
        buf.m.planes = planes;
        buf.length = num_planes;
    }
    if (xioctl(fd, VIDIOC_DQBUF, &buf) < 0)
        return -errno;
    *index = buf.index;
    if (is_mplane(type)) {
        for (unsigned int i = 0; i < num_planes; i++) {
            bytesused[i] = planes[i].bytesused;
            data_offset[i] = planes[i].data_offset;
        }
    } else {
        bytesused[0] = buf.bytesused;
        data_offset[0] = 0;
    }
    return 0;
}

// v4l2_stream_on starts streaming.
int v4l2_stream_on(int fd, unsigned int type) {
    enum v4l2_buf_type t = type;
    if (xioctl(fd, VIDIOC_STREAMON, &t) < 0)
        return -errno;
    return 0;
}

// v4l2_stream_off stops streaming.
int v4l2_stream_off(int fd, unsigned int type) {
    enum v4l2_buf_type t = type;
    if (xioctl(fd, VIDIOC_STREAMOFF, &t) < 0)
        return -errno;
    return 0;
}

// v4l2_poll waits for a frame to be ready. Returns >0 if ready, 0 on timeout.
int v4l2_poll(int fd, int timeout_ms) {
    struct pollfd pfd;
    pfd.fd = fd;
    pfd.events = POLLIN;
    int r;
    do {
        r = poll(&pfd, 1, timeout_ms);
    } while (r < 0 && errno == EINTR);
    if (r < 0) return -errno;
    return r;
}

// Constants exported for Go.
unsigned int v4l2_cap_video_capture(void) { return V4L2_CAP_VIDEO_CAPTURE; }
unsigned int v4l2_cap_video_capture_mplane(void) { return V4L2_CAP_VIDEO_CAPTURE_MPLANE; }
unsigned int v4l2_cap_streaming(void) { return V4L2_CAP_STREAMING; }
unsigned int v4l2_buf_type_capture(void) { return V4L2_BUF_TYPE_VIDEO_CAPTURE; }
unsigned int v4l2_buf_type_capture_mplane(void) { return V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE; }
unsigned int v4l2_max_planes(void) { return VIDEO_MAX_PLANES; }
