#ifndef C_UTILS_H
#define C_UTILS_H

#include <stdlib.h>
#include <errno.h>
#include <sys/stat.h>
#include <stdio.h>
#include <string.h>

/* Create directory and parents (like mkdir -p). Extern to be callable from
	other translation units. Returns EXIT_SUCCESS on success, EXIT_FAILURE on
	error, or EEXIST if a non-directory exists at the path. */
int _mkdir(const char *, mode_t);

#endif /* C_UTILS_H */