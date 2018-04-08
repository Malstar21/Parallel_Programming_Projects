// Parallel merge sort with OpenMP

#include <iostream>
#include <cstdlib> // for atoi()
#include <stdlib.h> // for srand, rand, for qsort
#include <time.h> // for time
#include <omp.h>  // openmp library
using namespace std;

/*
  Compiled using windows Cygwin
  g++ -fopen filename.cpp

  This program will create an n sized list of random values from 0 <= n and p number
  of threads.

  Each thread will be responsible for taking a part of the list of values into a
  local list of values that will then be quick sorted and inserted back into
  the global list.

  After words, each thread will either send or receive there localList they sorted
  if sender thread won't do anything if receiving thread will get sender thread's
  localList location and perform the merge operation.

  n can be any size since the last thread to be created will handle the remaining
  values to be quick sorted and sent. It will extend it's reach to the remaining values.
*/

// function prototypes
int compare (const void * a, const void * b);
void localSort(int n, int p, int* globalList, int remain);
void merge(int receiver, int sender, int listSize, int coreDiff, int divisor, int* globalList, int n, int p);
void checkSort(int* globalList, int n);

int main(int argc,char* argv[]) {
  int n = atoi(argv[1]);  // get n length
  int p = atoi(argv[2]);  // get total number of threads
  cout << "Using P=" << p << ", N=" << n << endl;
  int remain = 0;

  // check if n & p is evenly disvable
  if((n%p) != 0)
    remain = n%p;

  double startTime, endTime;
  int* listNum = new int[n];

  srand(time(NULL));  // seed srand
  #pragma parallel for
    for(int i = 0; i < n; i++) {
        listNum[i] = rand() % n; // range from 0 <= n
    }

  // start time
  startTime = omp_get_wtime();
  #pragma omp parallel num_threads(p)
    localSort(n,p,listNum,remain);

  // get end time
  endTime = omp_get_wtime();

  // display listNum array
  for(int j = 0; j < n; j++) {
    cout << listNum[j] << " ";
  }
  cout << endl;

  // display time
  cout << "TIME:" << endTime - startTime << endl;

  // check if sorted
  checkSort(listNum, n);
  return 0;
}

// helper function for quick sort
int compare (const void * a, const void * b) {
  return ( *(int*)a - *(int*)b );
}

// This function will handle quickSort and will call the merge
// function if thread is receiver
// function will create a local list of values that will be quick sorted
// then updated back into the global list, then merge will be called
// to merge the sender and receiver threads.
// Before calling merge will have a barrier preventing threads
// from going steps ahead before merge.
void localSort(int n, int p, int* globalList, int remain) {
  int my_rank = omp_get_thread_num(); // get thread rank
  int listSize, my_start, my_end;

  // if last thread running and we have remaining values add to
  // list size. listSize and my_start will include remaining value
  if(remain != 0 && my_rank == p - 1) {
    listSize = n/p + remain;
    my_start = (listSize-remain)*my_rank;  // get start in global
  }
  else {  // otherwise listSize set to n/p & my_start set to listSize*my_rank
    listSize = n/p;
    my_start = listSize*my_rank;  // get start in global
  }

  int* localList = new int[listSize]; // create localList
  my_end = my_start + listSize; // get end in global

  int i = 0;
  int tempStart = my_start; // set temp value to be used for loop

  // pass values to localList from globalList
  for(tempStart; tempStart < my_end; tempStart++, i++) {
    localList[i] = globalList[tempStart];
  }

  // display local list before quick sort
  // used critical here for display purpose
  #pragma omp critical
  {
    cout << "Thread: "<< my_rank << " local_list: ";
    for(int x = 0; x < listSize; x++) {
      cout << localList[x] << ", ";
    }
    cout << endl;
  }

  // quick sort localList
  qsort(localList, listSize, sizeof(int), compare);

  // display local list after quick sort
  // used critical here for display purpose
  #pragma omp critical
  {
    cout << "Thread: "<<  my_rank << " Quick Sorted local_list: ";
    for(int x = 0; x < listSize; x++) {
      cout << localList[x] << ", ";
    }
    cout << endl;
  }

  // update globalList with localList values
  i = 0;  // reset i to 0 to be reused
  #pragma parallel for
    for (my_start; my_start < my_end; my_start++, i++) {
    globalList[my_start] = localList[i];
  }

  int divisor = 2;
  int coreDifference = 1;

  while(divisor <= p) {
    #pragma omp barrier // openmp barrier for threads
    if(my_rank % divisor == 0) {  //  receiver
      // call merge sort
      merge(my_rank, my_rank + coreDifference, listSize, coreDifference, divisor, globalList, n, p);
    }
    else {  // sender
      // do nothing
    }

    divisor *= 2;
    coreDifference *= 2;
  }
}


// Merge will take the receiver and sender location start and end values in global array
// and merge sender with receiver list as well as check if sender has any remaining values
// that need to be added during the merge.
// whichever sender, or receiver values finish merge first the remaining list will be merged without
// any comparison.
void merge(int receiver, int sender, int listSize, int coreDiff, int divisor, int* globalList, int n, int p) {
  int recStart = receiver * listSize;
  int recEnd = recStart + (listSize * coreDiff);
  int sendStart = sender * listSize;
  int sendEnd = sendStart + (listSize * coreDiff);
  int size = listSize*divisor;

  // Will check if sender thread contains remaining values to be merged
  // if remaining values true, will add n%p to sendEnd and size
  // that will allow the receiver thread to merge those values
  // allows for any sized n
  if(sender == (p - coreDiff) && (n%p) != 0) {
    sendEnd += n%p; // add to sender's end size
    size += n%p;  // add to size
  }

  int* localList = new int[size]; // create new localList
  int i = 0;

  int tempRecStart = recStart;  // temp reciever start
  int tempSendStart = sendStart;  // temp sender start

  // while there exist values in both threads that have not been merged
  // contiune to merge values from both sender and reciever
  while(tempRecStart != recEnd && tempSendStart != sendEnd) {
    if(globalList[tempRecStart] <= globalList[tempSendStart]) {
      localList[i] = globalList[tempRecStart];
      i++;
      tempRecStart++;
    }
    else if(globalList[tempRecStart] >= globalList[tempSendStart]) {
      localList[i] = globalList[tempSendStart];
      i++;
      tempSendStart++;
    }
  }

  // If sender or receiver has finished merging then merge rest of
  // values from either sender or receiver.

  // if sender still has values to merge, merge them
  // else if receiver has values to merge, merge them
  if(tempRecStart == recEnd && tempSendStart != sendEnd) {
    for(tempSendStart; tempSendStart < sendEnd; tempSendStart++, i++) {
      localList[i] = globalList[tempSendStart];
    }
  }
  else if(tempSendStart == sendEnd && tempRecStart != recEnd) {
    for(tempRecStart; tempRecStart < recEnd; tempRecStart++, i++) {
      localList[i] = globalList[tempRecStart];
    }
  }

  // update globalList
  i = 0;  // reset i to be reused
  #pragma parallel for
    for(recStart; recStart < sendEnd; recStart++, i++) {
      globalList[recStart] = localList[i];
  }
}

// function will check final global list to make sure it is sorted
// runs at end of program
// displays SORTED, or NOT SORTED
void checkSort(int* globalList, int n) {
  for(int i = 0; i < n - 1; i++) {
    if(globalList[i] > globalList[i+1]) {
      cout << "NOT SORTED" << endl;
      return;
    }
  }
  cout << "SORTED" << endl;
}
