#include <stdint.h>

// ort_vcall calls a C function pointer with up to 8 pointer-sized args.
// All ORT API functions take pointer/size_t args and return OrtStatus*.
// Extra args beyond what the callee reads are harmless (sit in registers).
uintptr_t ort_vcall(uintptr_t fn,
    uintptr_t a0, uintptr_t a1, uintptr_t a2, uintptr_t a3,
    uintptr_t a4, uintptr_t a5, uintptr_t a6, uintptr_t a7) {
    typedef uintptr_t (*fn_t)(uintptr_t, uintptr_t, uintptr_t, uintptr_t,
                              uintptr_t, uintptr_t, uintptr_t, uintptr_t);
    return ((fn_t)fn)(a0, a1, a2, a3, a4, a5, a6, a7);
}

// ort_vcall_void is for Release functions that return void.
void ort_vcall_void(uintptr_t fn, uintptr_t a0) {
    typedef void (*fn_t)(uintptr_t);
    ((fn_t)fn)(a0);
}
