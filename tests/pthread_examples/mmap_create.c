#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <complex.h>
#include <stdint.h>

#define FILEPATH "/tmp/mmapped.bin"
#define NUMINTS  (1<<26)
#define FILESIZE (NUMINTS * sizeof(double complex))

int main(int argc, char *argv[])
{
    uint64_t i;
    int fd;
    int result;
    double complex *map;  /* mmapped array of complex's */
    
    printf("page size: %lu\n", (size_t) sysconf (_SC_PAGESIZE));

    /* Open a file for writing.
     *  - Creating the file if it doesn't exist.
     *  - Truncating it to 0 size if it already exists. (not really needed)
     *
     * Note: "O_WRONLY" mode is not sufficient when mmaping.
     */
    fd = open(FILEPATH, O_RDWR | O_CREAT | O_TRUNC, (mode_t)0600);
    if (fd == -1) {
	perror("Error opening file for writing");
	exit(EXIT_FAILURE);
    }

    /* Stretch the file size to the size of the (mmapped) array of ints
     */
    result = lseek(fd, FILESIZE-1, SEEK_SET);
    if (result == -1) {
	close(fd);
	perror("Error calling lseek() to 'stretch' the file");
	exit(EXIT_FAILURE);
    }
    
    /* Something needs to be written at the end of the file to
     * have the file actually have the new size.
     * Just writing an empty string at the current file position will do.
     *
     * Note:
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

    /* Now the file is ready to be mmapped.
     */
    map =  (double complex*) mmap(0, FILESIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (map == MAP_FAILED) {
	close(fd);
	perror("Error mmapping the file");
	exit(EXIT_FAILURE);
    }
    
    
    printf("page size (after mmaping): %lu\n", (size_t) sysconf (_SC_PAGESIZE));
    
    /* Now write int's to the file as if it were memory (an array of ints).
     */
    for (i = 0; i <NUMINTS; ++i) {
	map[i] = 2 * i + I; 
    }

    /* Don't forget to free the mmapped memory
     */
    if (munmap(map, FILESIZE) == -1) {
	perror("Error un-mmapping the file");
	/* Decide here whether to close(fd) and exit() or not. Depends... */
    }

    /* Un-mmaping doesn't close the file, so we still need to do that.
     */
    close(fd);
    return 0;
}