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
    double complex *map;  /* mmapped array of int's */
    
    printf("PROT_READ: %d\n", PROT_READ );
    printf("PROT_WRITE: %d\n", PROT_WRITE );
    printf("PROT_READ | PROT_WRITE: %d\n", PROT_READ | PROT_WRITE );

    fd = open(FILEPATH, O_RDONLY);
    if (fd == -1) {
	perror("Error opening file for reading");
	exit(EXIT_FAILURE);
    }

    map = (double complex*) mmap(0, FILESIZE, PROT_READ, MAP_SHARED, fd, 0);
    if (map == MAP_FAILED) {
	close(fd);
	perror("Error mmapping the file");
	exit(EXIT_FAILURE);
    }
    
    /* 
     * Read the file int-by-int from the mmap
     */
    for (i = 0; i <NUMINTS; ++i) {
      if (map[i] != 2 * i + I)
	printf("%lu: %lf + %lfj\n", i, creal(map[i]), cimag(map[i]));
    }

    if (munmap(map, FILESIZE) == -1) {
	perror("Error un-mmapping the file");
    }
    close(fd);
    return 0;
}