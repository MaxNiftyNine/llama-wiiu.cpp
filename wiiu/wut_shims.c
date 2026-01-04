#include <errno.h>
#include <malloc.h>
#include <unistd.h>

// Minimal shims for libc calls that are missing from CafeOS.
long sysconf(int name) {
    switch (name) {
        case _SC_PHYS_PAGES:
            // Approximate total pages; Wii U has ~1 GiB available to titles, but be conservative.
            return (512 * 1024 * 1024) / 4096;
        case _SC_PAGE_SIZE:
            return 4096;
        default:
            return -1;
    }
}

int posix_memalign(void **memptr, size_t alignment, size_t size) {
    void *p = memalign(alignment, size);
    if (!p) {
        return errno ? errno : ENOMEM;
    }
    *memptr = p;
    return 0;
}
