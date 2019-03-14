/*  Vector addition on the GPU: C = A + B  */
#include <stdio.h>
#include <stdlib.h>
#include <time.h> 
#include <sys/time.h> 
#include <omp.h>

// For consistency with GPU implementation
#define SIZE 1000000
double timer = 0.0;

// CPU version of the vector addition function
void vecAddCPU(float * A, float * B, float * C, int N)
{

   int i;
   #pragma omp parallel for default(shared) schedule(static)
   for (i=0; i<N; i++)
   {
      C[i] = A[i] + B[i];
   }

}

void timeit(int mode)
{

   static struct timeval tv[2];

   gettimeofday(&tv[mode],NULL); // from sys/time.h
   if ( mode )
      timer +=
         (double) ( tv[1].tv_usec - 
                    tv[0].tv_usec  ) / 1000 +
         (double) ( tv[1].tv_sec - 
                    tv[0].tv_sec);
}

/* Host function for filling vector (1d array) with 
   random numbers between -20.0 and 20.0 */
void fillOutVector( float * vec, int vec_length )
{

   time_t t;
   srand((unsigned) time(&t)); // initialize random number generator
   int i;
   for (i=0; i<vec_length; i++)
   {
      vec[i] = ( (float)rand() / (float)(RAND_MAX) ) * 40.0;
      vec[i] -= 20.0;
   }

}

// Host function for printing a vector (1d array)
void printVector( float * vec, int vec_length )
{
   int i;
   for (i=0; i<vec_length; i++) {
      printf("i: %d vec[i]: %f\n",i,vec[i]);
   }

}

// program execution begins here
int main( int argc, char ** argv )
{

   size_t vec_bytes = SIZE * sizeof(float);

   // host arrays
   float * h_A = (float *)malloc( vec_bytes );
   float * h_B = (float *)malloc( vec_bytes );
   float * h_C = (float *)malloc( vec_bytes );

   // fill array with random floats
   fillOutVector( h_A, SIZE );
   fillOutVector( h_B, SIZE );

   // compute the sum of vector A and B on CPU
   float * gold_C = (float *)malloc( vec_bytes );
   timeit(0);
   vecAddCPU( h_A, h_B, gold_C, SIZE );
   timeit(1);
   printf("CPU vector add: %10.3f ms\n",timer);
   printf("effective host bandwidth (GB/s): %5.3f\n",
          ( 3.0f * 4.0f * (float)SIZE / 1.0e9 ) / 
          ( timer / 1.0e3 ) );

   // print result of vector addition
   //printVector( gold_C, SIZE );

   // free memory on host
   free(h_A);
   free(h_B);
   free(h_C);
   free(gold_C);

   return 0;
}