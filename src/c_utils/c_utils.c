#include "c_utils.h"

// Source - https://stackoverflow.com/a
// Posted by Carl Norum, modified by community. See post 'Timeline' for change history
// Modified by Colin Pannikkat to include error handling and messages.
// Retrieved 2025-11-14, License - CC BY-SA 4.0
int _mkdir(const char *dir, mode_t mode) {

    struct stat st = {0};
    if (stat(dir, &st) == 0) {
        printf("Directory exists.");
        return EEXIST;
    }

    char tmp[256];
    char *p = NULL;
    size_t len;

    snprintf(tmp, sizeof(tmp), "%s", dir);
    len = strlen(tmp);
    if (len == 0)
        return EXIT_FAILURE;
    if (tmp[len - 1] == '/')
        tmp[len - 1] = 0;
    for (p = tmp + 1; *p; p++) {
        if (*p == '/') {
            *p = 0;
            if (mkdir(tmp, mode) != 0) {
                if (errno != EEXIST) {
                    perror("Failed to create directory");
                    return EXIT_FAILURE;
                }
            }
            *p = '/';
        }
    }
    if (mkdir(tmp, mode) != 0) {
        if (errno != EEXIST) {
            perror("Failed to create directory");
            return EXIT_FAILURE;
        }
    }
    printf("Directory successfully created.\n");
    return EXIT_SUCCESS;
}