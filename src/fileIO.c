#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <stdint.h>
#include <complex.h>

#include "global.h"
#include "fileIO.h"


const uint8_t filename_str_lenght = 128;

/*
 * In case to have transparent code - open files in special function and store pointers to the files in an array
 * PRZEMYSLEC ILE PLIKOW POTRZEBA -> WAVEFUNCTION, WCZYTYWANIE, BACKUP, ENERGIA, PRZEKROJE
 * CZYTAC/ZAPISYWAC WAVEFUNCTION DO PLIKOW BINARNYCH ZA POMOCA MMAP, A TIMING JAKOS INACZEJ (DO .TXT LUB nvprof UZYWAC)
 */
FILE** open_files() {
  
  const uint8_t num_files = 3;
  
  FILE** files = (FILE**) malloc( num_files*sizeof(FILE*) );
  
  char wf_filename[filename_str_lenght];
  FILE* wf_file = NULL;
  
  
  // move to creating files
  char backup_filename[filename_str_lenght];
  FILE* backup_file = NULL;
  sprintf(backup_filename,"./backup_dim%d_N%d.txt", DIM, NX*NY*NZ);
  printf("backup save in: %s\n",backup_filename);
  backup_file = fopen(backup_filename,"w");
  if (!backup_file) printf("Error opening file %s!\n",backup_filename);
  
  files[2] = backup_file; // enum -> BACKUP_FILE
  
  
  
  for (uint8_t ii=0; ii< num_files; ii++) {
    //files[ii] = fopen / mmap
    
  }
  
  
  return files;
}


/*
 * Closes files form array of pointers to files.
 * FILE** files - array of pointers to files
 *  const uint8_t num_files - number of files in the array
 */
void close_files(FILE** files, const uint8_t num_files) {
  for (uint8_t ii = 0; ii< num_files; ii++)
    if (files[ii]) fclose(files[ii]);
}


/*
 * Creates memory map from file. It can be used for reading/writing from big files, so for wavefunctions.
 * Typical size of 3D wavefunction is about 60MB - 2GB (in a fixed point in time!)
 * char* filepath - path to a file to be mmapped
 * size_t filesize - size of memory block to be mmapped
 * int protect - modes: PROT_READ | PROT_WRITE
 * int flags - like in documantation of mmap
 */
int mmap_create(char* filepath, void** map, size_t filesize, int protect, int flags) {
  
  int fd;
  
  fd = open(filepath, O_RDWR);
  if (fd == -1) {
    perror("Error opening file for reading");
    exit(EXIT_FAILURE);
  }
  
  *map = mmap(0, filesize, protect, flags, fd, 0);
  if (map == MAP_FAILED) {
    close(fd);
    perror("Error mmapping the file");
    exit(EXIT_FAILURE);
  }
#ifdef DEBUG
  if (protect == PROT_READ) 
    printf("\n\t\t\t\tenabled only reading to file %s\n", filepath);
#endif
  if (protect & PROT_WRITE) {
    printf("enabled writing to file %s\n", filepath);
    
  }
  return fd;
}


void mmap_destroy(int fd, void* map, size_t filesize) {
  if (fd != -1) {
    if (munmap(map, filesize) == -1) {
      perror("Error un-mmapping the file");
      /* Decide here whether to close(fd) and exit() or not. Depends... */
    }
    
    /* Un-mmaping doesn't close the file, so we still need to do that.
    */
    close(fd);
  }
}
