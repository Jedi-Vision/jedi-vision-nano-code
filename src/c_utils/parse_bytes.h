#ifndef C_UTILS_PARSE_BYTES_H
#define C_UTILS_PARSE_BYTES_H

#include <stddef.h>
#include <stdint.h>
#include "object.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Parse a buffer containing an object representation.
 * Returns 0 on success, negative on error.
 * buf and out must not be NULL. len is the size of buf in bytes.
 */
int parse_object_rep(const uint8_t *buf, size_t len, ObjectRepData *out);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* C_UTILS_PARSE_BYTES_H */