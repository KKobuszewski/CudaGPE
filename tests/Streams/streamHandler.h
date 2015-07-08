#ifndef __STREAMHANDLER_H__
#define __STREAMHANDLER_H__

typedef struct __attribute__((packed)) StreamsArray {
  cudaStream_t** streams_ptr_arr;
  uint8_t num_streams;
  
} StreamsArray;

#endif