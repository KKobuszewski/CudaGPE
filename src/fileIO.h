#ifndef __FILEIO_H__
#define __FILEIO_H__

// types definitions and enums

/*
 * TODO:
 * - replace files with this structure
 */
enum file_modes {BINARY,TEXT};
typedef struct _FILE {
    uint16_t file_index;// index of file
    char filename[128];     // name of file
    uint8_t mode;       // this flag will tell if this file is alble to write in text mode or in binary mode
    int permissions;    // permissions (read / write mode)
    FILE* data;         // pointer to file
} struct_file;

typedef struct _MEMMAP {
    
} struct_mmap;



// functions' definitions
FILE** open_files();
struct_file** open_struct_files(const uint8_t num_files);
void close_files(FILE** files, const uint8_t num_files);
void close_struct_files(struct_file** files, const uint8_t num_files);

// memory maps` utility functions
int mmap_create(char* filepath, void** map, size_t filesize, int protect, int flags);
void mmap_destroy(int fd, void* map, size_t filesize);



#endif