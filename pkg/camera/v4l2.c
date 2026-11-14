//go:build cgo

#include <linux/videodev2.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <poll.h>
#include <string.h>
#include <errno.h>

// v4l2_open opens a video device.
int v4l2_open(const char *path) {
    return open(path, O_RDWR | O_NONBLOCK);
}

// v4l2_close closes a file descriptor.
int v4l2_close(int fd) {
    return close(fd);
}

// v4l2_querycap fills name (up to name_len) and returns capabilities.
// Returns 0 on success, -errno on failure.
int v4l2_querycap(int fd, char *name, int name_len, unsigned int *caps) {
    struct v4l2_capability cap;
    memset(&cap, 0, sizeof(cap));
    if (ioctl(fd, VIDIOC_QUERYCAP, &cap) < 0)
        return -errno;
    int n = strlen((char *)cap.card);
    if (n >= name_len) n = name_len - 1;
    memcpy(name, cap.card, n);
    name[n] = 0;
    *caps = cap.capabilities;
    return 0;
}

// v4l2_set_format sets the capture format.
// Returns 0 on success, -errno on failure. actual_pixfmt is filled with
// the format the driver actually selected.
int v4l2_set_format(int fd, unsigned int pixfmt, unsigned int w, unsigned int h,
                    unsigned int *actual_pixfmt, unsigned int *actual_w, unsigned int *actual_h) {
    struct v4l2_format fmt;
    memset(&fmt, 0, sizeof(fmt));
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width = w;
    fmt.fmt.pix.height = h;
    fmt.fmt.pix.pixelformat = pixfmt;
    fmt.fmt.pix.field = V4L2_FIELD_NONE;
    if (ioctl(fd, VIDIOC_S_FMT, &fmt) < 0)
        return -errno;
    *actual_pixfmt = fmt.fmt.pix.pixelformat;
    *actual_w = fmt.fmt.pix.width;
    *actual_h = fmt.fmt.pix.height;
    return 0;
}

// v4l2_request_buffers requests mmap buffers. count is in/out.
int v4l2_request_buffers(int fd, unsigned int *count) {
    struct v4l2_requestbuffers req;
    memset(&req, 0, sizeof(req));
    req.count = *count;
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;
    if (ioctl(fd, VIDIOC_REQBUFS, &req) < 0)
        return -errno;
    *count = req.count;
    return 0;
}

// v4l2_query_buffer queries a buffer and returns its offset and length for mmap.
int v4l2_query_buffer(int fd, unsigned int index, unsigned int *offset, unsigned int *length) {
    struct v4l2_buffer buf;
    memset(&buf, 0, sizeof(buf));
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;
    buf.index = index;
    if (ioctl(fd, VIDIOC_QUERYBUF, &buf) < 0)
        return -errno;
    *offset = buf.m.offset;
    *length = buf.length;
    return 0;
}

// v4l2_mmap maps a buffer.
void *v4l2_mmap(int fd, unsigned int offset, unsigned int length) {
    void *p = mmap(NULL, length, PROT_READ | PROT_WRITE, MAP_SHARED, fd, offset);
    if (p == MAP_FAILED) return NULL;
    return p;
}

// v4l2_munmap unmaps a buffer.
int v4l2_munmap(void *addr, unsigned int length) {
    return munmap(addr, length);
}

// v4l2_queue_buffer enqueues a buffer.
int v4l2_queue_buffer(int fd, unsigned int index) {
    struct v4l2_buffer buf;
    memset(&buf, 0, sizeof(buf));
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;
    buf.index = index;
    if (ioctl(fd, VIDIOC_QBUF, &buf) < 0)
        return -errno;
    return 0;
}

// v4l2_dequeue_buffer dequeues a buffer. Returns the buffer index via *index
// and the number of bytes used via *bytesused. Returns -EAGAIN if no buffer
// is ready (non-blocking).
int v4l2_dequeue_buffer(int fd, unsigned int *index, unsigned int *bytesused) {
    struct v4l2_buffer buf;
    memset(&buf, 0, sizeof(buf));
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;
    if (ioctl(fd, VIDIOC_DQBUF, &buf) < 0)
        return -errno;
    *index = buf.index;
    *bytesused = buf.bytesused;
    return 0;
}

// v4l2_stream_on starts streaming.
int v4l2_stream_on(int fd) {
    enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (ioctl(fd, VIDIOC_STREAMON, &type) < 0)
        return -errno;
    return 0;
}

// v4l2_stream_off stops streaming.
int v4l2_stream_off(int fd) {
    enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (ioctl(fd, VIDIOC_STREAMOFF, &type) < 0)
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
unsigned int v4l2_cap_streaming(void) { return V4L2_CAP_STREAMING; }
unsigned int v4l2_pix_fmt_mjpeg(void) { return V4L2_PIX_FMT_MJPEG; }
unsigned int v4l2_pix_fmt_yuyv(void) { return V4L2_PIX_FMT_YUYV; }
