/*
 *  compile: g++ -o pthread_barier_example pthread_barier_example.c -lm -lpthread
 */

#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <stdlib.h>
#include <time.h>
#include <pthread.h>
//#include <sys/neutrino.h>

pthread_barrier_t   barrier; // barrier synchronization object
pthread_mutex_t mutex; // mutex
volatile char flag = 1;

void *
thread1 (void *not_used)
{
    time_t  now;

    time (&now);
    printf ("thread1 starting at %s", ctime (&now));

    // do the computation
    for (int ii=0; ii < 10; ii++) {
        
        //sleep (20);
        now = clock();
        pthread_barrier_wait (&barrier);
        //pthread_mutex_lock (&mutex);
        if (ii == 9) flag = 0;
        //pthread_mutex_unlock (&mutex);
        pthread_barrier_wait (&barrier);
        printf("waiting %.5f ms\n",(clock() - now)*1000./CLOCKS_PER_SEC);
        // after this point, all three threads have completed.
        time (&now);
        printf ("%i. barrier in thread1() done at %s", ii, ctime (&now));
        printf("flag: %d\n",flag);
        
    }
}

void *
thread2 (void *not_used)
{
    time_t  now;

    time (&now);
    printf ("thread2 starting at %s", ctime (&now));
    
    int ii = 0;
    volatile char stop_cond = 1;
    while(stop_cond) {
        now = clock();
        //if (!flag) break;
        pthread_barrier_wait (&barrier);
        pthread_barrier_wait (&barrier);
        //pthread_mutex_lock (&mutex);
        stop_cond = flag;
        //pthread_mutex_unlock (&mutex);
        printf("waiting %.5f ms\n",(clock() - now)*1000./CLOCKS_PER_SEC);
        time (&now);
        printf ("%i. barrier in thread2() done at %s", ii, ctime (&now));
        printf("flag: %d\n",flag);
        ii++;
    } 
    
    /*
     * This is OK:
    for (int ii=0; ii < 10; ii++) {
        
        pthread_mutex_lock (&mutex);
        //if (!flag) break;
        pthread_mutex_unlock (&mutex);
        // do the computation
        // let's just do a sleep here...
        //sleep (40);
        pthread_barrier_wait (&barrier);
        // after this point, all three threads have completed.
        time (&now);
        printf ("%i. barrier in thread2() done at %s", ii, ctime (&now));
    }
    */
}

int main () // ignore arguments
{
    time_t  now;
    cpu_set_t cpu_core;

    // create a barrier object with a count of 3
    pthread_barrier_init (&barrier, NULL, 2);
    pthread_t th1;
    pthread_t th2;
    // start up two threads, thread1 and thread2
    pthread_create (&th1, NULL, thread1, NULL);
    pthread_create (&th2, NULL, thread2, NULL);
    
    CPU_ZERO(&cpu_core);
    CPU_SET(0, &cpu_core);
    pthread_setaffinity_np(th1, sizeof(cpu_set_t), &cpu_core);
    
    CPU_SET(1, &cpu_core);
    pthread_setaffinity_np(th2, sizeof(cpu_set_t), &cpu_core);
    // at this point, thread1 and thread2 are running

    // now wait for completion
    /*time (&now);
    printf ("main() waiting for barrier at %s", ctime (&now));
    pthread_barrier_wait (&barrier);

    // after this point, all three threads have completed.
    time (&now);
    printf ("barrier in main() done at %s", ctime (&now));*/
    pthread_exit( NULL );
    return (EXIT_SUCCESS);
}