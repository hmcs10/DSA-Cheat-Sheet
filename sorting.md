# Selection Sort:
## Approach->
Selection sort is a simple and efficient sorting algorithm that works by repeatedly selecting the smallest (or largest) element from the unsorted portion of the list and moving it to the sorted portion of the list. 
Lets consider the following array as an example: arr[] = {64, 25, 12, 22, 11}

First pass:

For the first position in the sorted array, the whole array is traversed from index 0 to 4 sequentially. The first position where 64 is stored presently, after traversing whole array it is clear that 11 is the lowest value.
Thus, replace 64 with 11. After one iteration 11, which happens to be the least value in the array, tends to appear in the first position of the sorted list.

New array after first pass: arr[] = {11, 25, 12, 22, 64}

Second Pass:

For the second position, where 25 is present, again traverse the rest of the array in a sequential manner.
After traversing, we found that 12 is the lowest value in the array and it should appear at the second place in the array where i is, thus swap these values.

Continue this till the array is sorted

## Code ->
```cpp
for (int i = 0; i < n - 1; i++) {
    int mini = i;
    for (int j = i + 1; j < n; j++) {
      if (arr[j] < arr[mini]) {
        mini = j;
      }
    }
    swap(arr[i], arr[mini]);
}
```
Worst, Avg and Best TC -> O(n^2)

# Bubble Sort:
## Approach->
In Bubble Sort algorithm, 
traverse from left and compare adjacent elements, if they are not in the sorted order then swap them and continute doing that for one complete iteration and at the end the greatest element will automatically be placed at the end or placed at right side.
In the next iteration perform the same thing but till the second last index, that way we will get our second last greatest element at second last position. 
In this way, the largest element is moved to the rightmost end every time. 
This process is then continued to find the second largest and place it and so on until the data is sorted.

## Code->
```cpp
void bubbleSort(int arr[], int n)
{
    int i, j;
    bool swapped;
    for (i = 0; i < n - 1; i++) {
        swapped = false;
        for (j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                swap(arr[j], arr[j + 1]);
                swapped = true;
            }
        }
 
        // If no two elements were swapped by inner loop, that means the array is already completely sorted
        if (swapped == false)
            break;
    }
}
```
Worst and Avg TC -> O(n^2)
Best TC -> O(n)

# Insertion sort
## Approach ->
The idea is to insert the selected element at its correct position. 
- Select an element in each iteration from the unsorted array(using a loop).
- Place it in its corresponding position in the sorted part and shift the remaining elements accordingly (using an inner loop and swapping).
- The “inner while loop” basically shifts the elements using swapping.

## Code->
```cpp
for (int i = 0; i < n; i++) {
    int j = i;
    while (j > 0 && arr[j - 1] > arr[j]) {
        swap(arr[j-1], arr[j]);
        j--;
    }
}
```
TC-> O(n^2)

# Merge Sort

## Approach ->
- Merge Sort is a divide and conquers algorithm, it divides the given array into equal parts and then merges the 2 sorted parts. 
- There are 2 main functions :
-  merge(): This function is used to merge the 2 halves of the array. It assumes that both parts of the array are sorted and merges both of them.
-  mergeSort(): This function divides the array into 2 parts. low to mid and mid+1 to high where,

```
low = leftmost index of the array

high = rightmost index of the array

mid = Middle index of the array 
```

We recursively split the array, and go from top-down until all sub-arrays size becomes 1.

---
## Code ->
```cpp
#include <bits/stdc++.h>
using namespace std;

void merge(vector<int> &arr, int low, int mid, int high) {
    vector<int> temp; // temporary array
    int left = low;      // starting index of left half of arr
    int right = mid + 1;   // starting index of right half of arr

    //storing elements in the temporary array in a sorted manner

    while (left <= mid && right <= high) {
        if (arr[left] <= arr[right]) {
            temp.push_back(arr[left]);
            left++;
        }
        else {
            temp.push_back(arr[right]);
            right++;
        }
    }

    // if elements on the left half are still left 

    while (left <= mid) {
        temp.push_back(arr[left]);
        left++;
    }

    //  if elements on the right half are still left 
    while (right <= high) {
        temp.push_back(arr[right]);
        right++;
    }

    int x = 0;
    // transfering all elements from temporary to arr
    // Note 1
    for (int i = low; i <= high; i++) {
        arr[i] = temp[x++];
    }
    // instead of using i-low we could have created another variable x=0 and could have done temp[x++]
}

void mergeSort(vector<int> &arr, int low, int high) {
    //Note 2
    if (low >= high) return;
    int mid = (low + high) / 2 ;
    mergeSort(arr, low, mid);  // left half
    mergeSort(arr, mid + 1, high); // right half
    merge(arr, low, mid, high);  // merging sorted halves
}

int main() {

    vector<int> arr = {9, 4, 7, 6, 3, 1, 5}  ;
    int n = 7;

    cout << "Before Sorting Array: " << endl;
    for (int i = 0; i < n; i++) {
        cout << arr[i] << " "  ;
    }
    cout << endl;
    mergeSort(arr, 0, n - 1);
    cout << "After Sorting Array: " << endl;
    for (int i = 0; i < n; i++) {
        cout << arr[i] << " "  ;
    }
    cout << endl;
    return 0 ;
}
```

Note 1: Note how we transfer the values of temp into arr. We always have our unique low and high values, so we insert the values in arr from low to high. And for temp we use a variable x and keep inserting the values from x=0.

Note 2: In base case of mergeSort don't make the mistake of not returning null for low==high, because if we simply write if(low>high) then in that case this allows the condition l == r (i.e., one element) to still proceed into the recursion, which causes the midpoint to be the same as l, and the recursive calls never terminate properly for all cases. This will give you Segmentation Fault.

Time complexity: O(nlogn) 

Reason: At each step, we divide the whole array into two halves, for that logn and we assume n steps are taken to get sorted array, so overall time complexity will be nlogn

Space complexity: O(n)  

Reason: We are using a temporary array to store elements in sorted order.

# Quick Sort

## Approach/Intuition ->
Quick Sort is a divide-and-conquer algorithm like the Merge Sort. But unlike Merge sort, this algorithm does not use any extra array for sorting(though it uses an auxiliary stack space). So, from that perspective, Quick sort is slightly better than Merge sort.

This algorithm is basically a repetition of two simple steps that are the following:

Pick a pivot and place it in its correct place in the sorted array.
Shift smaller elements(i.e. Smaller than the pivot) on the left of the pivot and larger ones to the right.
Now, let’s discuss the steps in detail considering the array {4,6,2,5,7,9,1,3}:

Step 1: The first thing is to choose the pivot. A pivot is basically a chosen element of the given array. The element or the pivot can be chosen by our choice. So, in an array a pivot can be any of the following:

The first element of the array
The last element of the array
Median of array
Any Random element of the array
After choosing the pivot(i.e. the element), we should place it in its correct position(i.e. The place it should be after the array gets sorted) in the array. For example, if the given array is {4,6,2,5,7,9,1,3}, the correct position of 4 will be the 4th position.

Note: Here in this tutorial, we have chosen the first element as our pivot. You can choose any element as per your choice.

Step 2: In step 2, we will shift the smaller elements(i.e. Smaller than the pivot) to the left of the pivot and the larger ones to the right of the pivot. In the example, if the chosen pivot is 4, after performing step 2 the array will look like: {3, 2, 1, 4, 6, 5, 7, 9}. 

From the explanation, we can see that after completing the steps, pivot 4 is in its correct position with the left and right subarray unsorted. Now we will apply these two steps on the left subarray and the right subarray recursively. And we will continue this process until the size of the unsorted part becomes 1(as an array with a single element is always sorted).

So, from the above intuition, we can get a clear idea that we are going to use recursion in this algorithm.

To summarize, the main intention of this process is to place the pivot, after each recursion call, at its final position, where the pivot should be in the final sorted array.

## Code ->
```cpp
class Solution {
  public:
    // Function to sort an array using quick sort algorithm.
    void quickSort(vector<int>& arr, int low, int high) {
        // Base case: if the subarray has one or zero elements, it's already sorted.
        if(low >= high) return;
        
        // Partition the array and get the pivot index.
        int pivot = partition(arr, low, high);
        
        // Recursively sort the left subarray (elements less than the pivot).
        quickSort(arr, low, pivot - 1);
        // Recursively sort the right subarray (elements greater than the pivot).
        quickSort(arr, pivot + 1, high);
    }

  public:
    // Function that takes last element as pivot, places the pivot element at
    // its correct position in sorted array, and places all smaller elements
    // to left of pivot and all greater elements to right of pivot.
    int partition(vector<int>& arr, int low, int high) {
        // Choose the first element as the pivot.
        int pivot = low;
        int left = low;
        int right = high;
        
        // Loop to rearrange elements around the pivot.
        while(left <= right) {
            // Move the left pointer to the right until an element greater than the pivot is found.
            while(left <= right && arr[left] <= arr[pivot]) left++;
            // Move the right pointer to the left until an element less than or equal to the pivot is found.
            while(right >= left && arr[right] > arr[pivot]) right--;
            // If the left pointer is still less than the right pointer, swap the elements.
            if(left < right) swap(arr[left], arr[right]);
        }
        
        // Swap the pivot element with the element at the right pointer.
        swap(arr[low], arr[right]);
        
        // Return the index of the pivot element after partitioning.
        return right;
    }
};
```
- Time Complexity: O(N*logN), where N = size of the array.
- Space Complexity: O(1) + O(N) auxiliary stack space.