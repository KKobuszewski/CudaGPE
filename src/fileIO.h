#ifndef __FILEIO_H__
#define __FILEIO_H__

// functions' definitions
FILE** open_files();
void close_files(FILE** files, const uint8_t num_files);

// memory maps
int mmap_create(char* filepath, void** map, size_t filesize, int protect, int flags);
void mmap_destroy(int fd, void* map, size_t filesize);



#endif