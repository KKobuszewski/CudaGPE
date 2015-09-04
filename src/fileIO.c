#include <stdio.h>
#include <stdlib.h>
#include <time.h>
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

const char* files_names[] = {"simulation_parametrs","statistics","wf_frames","propagators","wf_k"};

/*
 * In case to have transparent code - open files in special function and store pointers to the files in an array
 * PRZEMYSLEC ILE PLIKOW POTRZEBA -> WAVEFUNCTION, WCZYTYWANIE, BACKUP, ENERGIA, PRZEKROJE
 * CZYTAC/ZAPISYWAC WAVEFUNCTION DO PLIKOW BINARNYCH ZA POMOCA MMAP, A TIMING JAKOS INACZEJ (DO .TXT LUB nvprof UZYWAC)
 */
struct_file** open_struct_files(const uint8_t num_files = 5) {
  
  char str_date[17];
  char dirname[256];
  
  struct_file** files = (struct_file**) malloc( num_files*sizeof(struct_file*) );
  for (uint8_t ii=0; ii < num_files; ii++) {
      files[ii] = (struct_file*) malloc( sizeof(struct_file) );
  }
  
  // files are going to be saved in special directory containing information of used potentials and date of simulation
  /*
   * TODO:  - some more information about simulation <- maybe .txt file with some more details
   *        - better way to preapre the name (str concat)
   * 
   */
  
  time_t t = time(NULL);
  strftime(str_date, sizeof(str_date), "%Y-%m-%d_%H:%M", localtime(&t));
#ifdef V_EXT
    #ifdef V_CON
    
        #ifdef V_DIP
        // V_EXT, V_CON, V_DIP are defined
        sprintf( dirname,"../data_TVextVcVd_dim%d_N%d_%s", DIM, NX*NY*NZ, str_date );
        #else
        // V_EXT, V_CON are defined
        sprintf( dirname,"../data_TVextVc_dim%d_N%d_%s", DIM, NX*NY*NZ, str_date );
        #endif
    
    #else
        // V_EXT is defined
        sprintf( dirname,"../data_TVext_dim%d_N%d_%s", DIM, NX*NY*NZ, str_date );
    #endif
#else
    #ifdef V_CON
    
        #ifdef V_DIP
        // V_CON, V_DIP are defined
        sprintf( dirname,"../data_TVcVd_dim%d_N%d_%s", DIM, NX*NY*NZ, str_date );
        #else
        // V_CON is defined
        sprintf( dirname,"../data_TVc_dim%d_N%d_%s", DIM, NX*NY*NZ, str_date );
        #endif
    
    #else
        // no potentials, internal or external, are defined
        sprintf( dirname,"../data_T_dim%d_N%d_%s", DIM, NX*NY*NZ, str_date );
    #endif
#endif
  
  // creating directory for simulation files
  struct stat st = {0};
  if (stat(dirname, &st) == -1) {
    mkdir(dirname, 0777);
  }
  printf("\nsaving data in directory: %s\n",dirname);
  
  //char* filenames[filename_str_lenght] = (char**) malloc( sizeof(char*)*num_files);
  for (uint8_t ii=0; ii<num_files; ii++) {
      
      if ( (ii == WF_FRAMES_FILE) || (ii == WF_K_FILE) ) {   // here open files for binary operations
          sprintf( (files[ii])->filename,"%s/%s_dim%d_N%d.bin", dirname, files_names[ii], DIM, NX*NY*NZ );
          files[ii]->data = fopen(files[ii]->filename,"wb");
          files[ii]->mode = BINARY;
          //files[ii]->permissions =
          printf("(binary mode) ");
      }
      else {
          sprintf( (files[ii])->filename,"%s/%s_dim%d_N%d.txt", dirname, files_names[ii], DIM, NX*NY*NZ );
          files[ii]->data = fopen(files[ii]->filename,"w");
          files[ii]->mode = TEXT;
          
          printf("(text mode) ");
      }
      
      if 
          (!files[ii]->data) printf("Error opening file %s!\n",files[ii]->filename);
      else 
          printf("%s save in: %s\n", files_names[ii],files[ii]->filename);
  }
  
  // linking with struct for global variables
  global_stuff->num_files = num_files;
  
  return files;
}



/*
 * In case to have transparent code - open files in special function and store pointers to the files in an array
 * PRZEMYSLEC ILE PLIKOW POTRZEBA -> WAVEFUNCTION, WCZYTYWANIE, BACKUP, ENERGIA, PRZEKROJE
 * CZYTAC/ZAPISYWAC WAVEFUNCTION DO PLIKOW BINARNYCH ZA POMOCA MMAP, A TIMING JAKOS INACZEJ (DO .TXT LUB nvprof UZYWAC)
 */
FILE** open_files() {
  
  const uint8_t num_files = 6;
  char str_date[17];
  char dirname[256];
  
  FILE** files = (FILE**) malloc( num_files*sizeof(FILE*) );
  
  // files are going to be saved in special directory containing information of used potentials and date of simulation
  /*
   * TODO:  - some more information about simulation <- maybe .txt file with some more details
   *        - better way to preapre the name (str concat)
   * 
   */
  
  time_t t = time(NULL);
  strftime(str_date, sizeof(str_date), "%Y-%m-%d_%H:%M", localtime(&t));  
#ifdef V_EXT
    #ifdef V_CON
    
        #ifdef V_DIP
        // V_EXT, V_CON, V_DIP are defined
        sprintf( dirname,"../data_TVextVcVd_dim%d_N%d_%s", DIM, NX*NY*NZ, str_date );
        #else
        // V_EXT, V_CON are defined
        sprintf( dirname,"../data_TVextVc_dim%d_N%d_%s", DIM, NX*NY*NZ, str_date );
        #endif
    
    #else
        // V_EXT is defined
        sprintf( dirname,"../data_TVext_dim%d_N%d_%s", DIM, NX*NY*NZ, str_date );
    #endif
#else
    #ifdef V_CON
    
        #ifdef V_DIP
        // V_CON, V_DIP are defined
        sprintf( dirname,"../data_TVcVd_dim%d_N%d_%s", DIM, NX*NY*NZ, str_date );
        #else
        // V_CON is defined
        sprintf( dirname,"../data_TVc_dim%d_N%d_%s", DIM, NX*NY*NZ, str_date );
        #endif
    
    #else
        // no potentials, internal or external, are defined
        sprintf( dirname,"../data_T_dim%d_N%d_%s", DIM, NX*NY*NZ, str_date );
    #endif
#endif
  
  // creating directory for simulation files
  struct stat st = {0};
  if (stat(dirname, &st) == -1) {
    mkdir(dirname, 0777);
  }
  printf("saving data in directory: %s",dirname);
  
  //char wf_filename[filename_str_lenght];
  //FILE* wf_file = NULL;
  
  
  // file for backup - really necessary???
  char backup_filename[filename_str_lenght];
  FILE* backup_file = NULL;
  sprintf(backup_filename,"./backup_dim%d_N%d.txt", DIM, NX*NY*NZ);
  printf("backup save in: %s\n",backup_filename);
  backup_file = fopen(backup_filename,"w");
  if (!backup_file) printf("Error opening file %s!\n",backup_filename);
  
  files[num_files-1] = backup_file; // enum -> BACKUP_FILE
  
  
  // file to save before starting algorithm
  char init_filename[filename_str_lenght];
  FILE* init_file = NULL;
  sprintf(init_filename,"./init_dim%d_N%d.txt", DIM, NX*NY*NZ);
  printf("init wf save in: %s\n",init_filename);
  init_file = fopen(init_filename,"w");
  if (!init_file) printf("Error opening file %s!\n",init_filename);
  
  files[0] = init_file;
  
  // file to save after FFT forward
  char FFT_filename[filename_str_lenght];
  FILE* FFT_file = NULL;
  sprintf(FFT_filename,"./FFT_dim%d_N%d.txt", DIM, NX*NY*NZ);
  printf("FFT wf save in: %s\n",FFT_filename);
  FFT_file = fopen(FFT_filename,"w");
  if (!FFT_file) printf("Error opening file %s!\n",FFT_filename);
  
  files[1] = FFT_file;
  
  
  // file to save after IFFT back
  char IFFT_filename[filename_str_lenght];
  FILE* IFFT_file = NULL;
  sprintf(IFFT_filename,"./IFFT_dim%d_N%d.txt", DIM, NX*NY*NZ);
  printf("IFFT wf save in: %s\n",IFFT_filename);
  IFFT_file = fopen(IFFT_filename,"w");
  if (!IFFT_file) printf("Error opening file %s!\n",IFFT_filename);
  
  files[2] = IFFT_file;
  
  // file to save stats of system
  char stats_filename[filename_str_lenght];
  FILE* stats_file = NULL;
  sprintf(stats_filename,"./stats_filename%d_N%d.txt", DIM, NX*NY*NZ);
  printf("stats wf save in: %s\n",stats_filename);
  stats_file = fopen(stats_filename,"w");
  if (!stats_file) printf("Error opening file %s!\n",stats_filename);
  
  files[3] = stats_file;
  
  // file to save propagator T
  char T_filename[filename_str_lenght];
  FILE* T_file = NULL;
  sprintf(T_filename,"./T_filename%d_N%d.txt", DIM, NX*NY*NZ);
  printf("stats wf save in: %s\n",T_filename);
  T_file = fopen(T_filename,"w");
  if (!T_file) printf("Error opening file %s!\n",T_filename);
  
  files[4] = T_file;
  
  
  // linking with struct for global variables <- wlasciwie to tylko komplikuje, ale chociaz wiadomo co jest global, a co nie ...
  //global_stuff->files = files;
  global_stuff->num_files = num_files;
  
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
 * Closes files form array of pointers to files.
 * FILE** files - array of pointers to files
 *  const uint8_t num_files - number of files in the array
 */
void close_struct_files(struct_file** files, const uint8_t num_files) {
  for (uint8_t ii = 0; ii< num_files; ii++)
    if ( (files[ii])->data ) fclose( (files[ii])->data );
  
  free( files );
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
  
  fd = open(filepath, O_RDWR|O_CREAT, S_IRUSR | S_IRGRP | S_IROTH);
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
    
    /* 
     * Stretch the file size to the size of the (mmapped) array of ints
     */
    long long int result = lseek(fd, filesize-1, SEEK_SET);
    if (result == -1) {
	close(fd);
	perror("Error calling lseek() to 'stretch' the file");
	exit(EXIT_FAILURE);
    }
    
    /* Something needs to be written at the end of the file to
     * have the file actually have the new size.
     * Just writing an empty string at the current file position will do.
     *
     * Note:, const uint8_t num_files
     *  - The current position in the file is at the end of the stretched 
     *    file due to the call to lseek().
     *  - An empty string is actually a single '\0' character, so a zero-byte
     *    will be written at the last byte of the file.
     */
    result = write(fd, "", 1);
    if (result != 1) {
	close(fd);
	perror("Error writing last byte of the file");
	exit(EXIT_FAILURE);
    }
    
    
    
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

