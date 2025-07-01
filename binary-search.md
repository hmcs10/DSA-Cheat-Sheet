# Implement Lower bound
## Question -> 
The lower bound algorithm finds the first or the smallest index in a sorted array where the value at that index is greater than or equal to a given key i.e. x.

The lower bound is the smallest index, ind, where arr[ind] >= x. But if any such index is not found, the lower bound algorithm returns n i.e. size of the given array.

Example-> arr[] = {1, 2, 4, 4, 4, 5, 9, 9}, x=4.

Output: 2 (as lower bound of 4 is at index 2. whereas upper bound of 4 is at index 5 as till index 4 there was 4 but at index 5 there is something greater than 4 (this will depend according to the question, don't worry about it so much))

---
## Approaches ->
- Naive approach (Using linear search)
- Optimal Approach (Using Binary Search): 
1. Place the 2 pointers i.e. low and high: Initially, we will place the pointers like this: low will point to the first index, and high will point to the last index.
2. Calculate the ‘mid’: Now, we will calculate the value of mid using the following formula:
`mid = (low+high) / 2`
3. Compare arr[mid] with x: With comparing arr[mid] to x, we can observe 2 different cases:
-  Case 1 – If arr[mid] >= x: This condition means that the index mid may be an answer. So, we will update the ‘ans’ variable with mid and search in the left half if there is any smaller index that satisfies the same condition. Here, we are eliminating the right half.
-  Case 2 – If arr[mid] < x: In this case, mid cannot be our answer and we need to find some bigger element. So, we will eliminate the left half and search in the right half for the answer.

## Code ->
```cpp
int lowerBound(vector<int> arr, int n, int x) {
    int low = 0, high = n - 1;
    int ans = n;

    while (low <= high) {
        int mid = (low + high) / 2;
        // maybe an answer
        if (arr[mid] >= x) {
            ans = mid;
            //look for smaller index on the left
            high = mid - 1;
        }
        else {
            low = mid + 1; // look on the right
        }
    }
    return ans;
}
```
- There's another way of doing this and that is using STL in cpp. 
- `lb = lower_bound(arr.begin(), arr.end(), x) - arr.begin();`
- This will return the index where there is lb.
- Here, x is the element we are searching the lb for. lower_bound(arr.begin(), arr.end(), x) will return an iterator hence we are subtracting it with arr.begin() to get the index value. So while solving problems you will most likely be using stl so remember it.

# Upper bound
Example -> arr[] = {1, 2, 4, 4, 4, 5, 9, 9}, x=4.

Output -> 5 (as at index 5 there is something greater than 4)

## Code ->
```cpp
int lowerBound(vector<int> arr, int n, int x) {
    int low = 0, high = n - 1;
    int ans = n;

    while (low <= high) {
        int mid = (low + high) / 2;
        // maybe an answer
        if (arr[mid] > x) { // Only this changed, rest everything is same as the code of lower bound
            ans = mid;
            //look for smaller index on the left
            high = mid - 1;
        }
        else {
            low = mid + 1; // look on the right
        }
    }
    return ans;
}
```

- STL: `ub = upper_bound(arr.begin(), arr.end(), x) - arr.begin();`

# [Floor and Ceil in Sorted Array](https://takeuforward.org/plus/dsa/problems/floor-and-ceil-in-sorted-array)
## Approach ->
Since the array is sorted, we can leverage Binary Search to efficiently find:

    Floor: The largest element ≤ x (if exists, else -1).

    Ceil: The smallest element ≥ x (if exists, else -1).

Instead of two separate passes, we optimize by tracking both floor and ceil in a single Binary Search pass, updating them based on comparisons with x. We will write the normal binary search but whenever we find an element greater than x we know that element might be our potential ceil. Similarly if we find an elem lower than x, that might be our potential floor. And if we found x, that element is surely our floor and ceil. 

```cpp
class Solution {
public:
    vector<int> getFloorAndCeil(vector<int>& nums, int x) {
        int floor = -1, ceil = -1;  // Initialize to -1 (not found)
        int low = 0, high = nums.size() - 1;

        while (low <= high) {
            int mid = low + (high - low) / 2;  // Avoid overflow

            if (nums[mid] == x) {  
                // Exact match → floor = ceil = x
                floor = ceil = nums[mid];
                break;
            }
            else if (nums[mid] < x) {  
                // Potential floor → store and search right
                floor = nums[mid];
                low = mid + 1;
            }
            else {  
                // Potential ceil → store and search left
                ceil = nums[mid];
                high = mid - 1;
            }
        }

        // Edge case: Ensure floor ≤ x and ceil ≥ x (handles duplicates)
        if (floor != -1 && floor > x) floor = -1;
        if (ceil != -1 && ceil < x) ceil = -1;

        return {floor, ceil};  // Return pair
    }
};
```

# [34. Find First and Last Position of Element in Sorted Array](https://leetcode.com/problems/find-first-and-last-position-of-element-in-sorted-array/description/)

## Approach ->
- This is a slight variation of the lower bound and upper bound problem. Here the approach is fairly simple, if we need to find the first positon of an element then use normal binary search but when you find the element we acutally need to update the answer but also check on the left side of the found position. So we will find the left side and in case of the last position when we find the element at any position we will only check on its right.

## Code ->
```cpp
class Solution {
public:
    int lowerBound(vector<int> nums, int target){
        int lb = -1, lo=0, hi=nums.size()-1, mid; // Initilize lb with -1 because if there is no ans then -1 will be returned.
         while(lo<=hi){
            mid = (lo+hi)/2;
            if(nums[mid]==target){ // if target is found
                lb = mid; // update the lb to be returned 
                hi = mid-1; // search on the left
            }
            else if(nums[mid]>target)
                hi = mid-1;
            else 
                lo = mid+1;
        }
        return lb;
    }
    int upperBound(vector<int> nums, int target){
        int ub = -1, lo=0, hi=nums.size()-1, mid; // Initilize ub with -1 because if there is no ans then -1 will be returned.
         while(lo<=hi){
            mid = (lo+hi)/2;
            if(nums[mid]==target){ // if target is found
                ub = mid; // update the lb to be returned 
                lo = mid+1; // search on the right
            }
            else if(nums[mid]>target)
                hi = mid-1;
            else 
                lo = mid+1;
        }
        return ub;
    }

    vector<int> searchRange(vector<int>& nums, int target) {
        vector<int> ans;
        ans.push_back(lowerBound(nums, target));
        ans.push_back(upperBound(nums, targt));

        return ans;
    }
};
```

# [33. Search in Rotated Sorted Array](https://leetcode.com/problems/search-in-rotated-sorted-array/description/)

## Approach ->
The approach is fairly simple once you actually understand it. To utilize the binary search algorithm effectively, it is crucial to ensure that the input array is sorted. By having a sorted array, we guarantee that each index divides the array into two sorted halves. In the search process, we compare the target value with the middle element, i.e. arr[mid], and then eliminate either the left or right half accordingly. This elimination becomes feasible due to the inherent property of the sorted halves(i.e. Both halves always remain sorted).

However, in this case, the array is both rotated and sorted. As a result, the property of having sorted halves no longer holds. This disruption in the sorting order affects the elimination process, making it unreliable to determine the target’s location by solely comparing it with arr[mid]. 

Key Observation: Though the array is rotated, we can clearly notice that for every index, one of the 2 halves will always be sorted.

So, to efficiently search for a target value using this observation, we will follow a simple two-step process:

- First, we identify the sorted half of the array. 
- Once found, we determine if the target is located within the bounds of this sorted half. 
-  If not, we eliminate that half from further consideration. 
-  Conversely, if the target does exist in the sorted half, we eliminate the other half.

It is necessary to first check which half is sorted so we can be completely sure that whether an element is falling within that half or not to make an informative decision whether to element that half or not.

## Code ->
```cpp
class Solution {
public:
    int search(vector<int>& nums, int target) {
        int lo = 0, hi = nums.size()-1, mid;

        while(lo <= hi){
            mid = (lo+hi)/2;
            if(nums[mid]==target) return mid; // if target found, return index

            if(nums[mid]>=nums[lo]){ // identified that left side is sorted
                if(nums[mid]>target && nums[lo]<=target) // if the target is between the sorted side
                    hi = mid-1;
                else lo = mid+1; // not between the sorted side
            }
            else{   // identifed that right side is sorted
                if(nums[mid]<target && nums[hi]>=target) // if the target is between the sorted side
                    lo = mid+1;
                else hi = mid-1; // not between the sorted side
            }
        }

        return -1;
    }
};
```
Note: `if(nums[mid]>=nums[lo])` In this check >= is important else there will be an edge case. eg of edge case 

```- [3,1]. Target = 1.```

# [81. Search in Rotated Sorted Array II](https://leetcode.com/problems/search-in-rotated-sorted-array-ii/description/)

## Approach->
The approach to this question will remain the same as of the question above but there is just one problem. The elements in the array can repeat. So imagine if our arry is [3, 0, 2, 3, 3, 3, 3] and target is 2. Mid will point to 3 but the algo will get confued that which part to prune because there is 3 at the starting position and also at the end position. Rest everything remains the same as that of the approach on the question above but we just need to tackle the above edge case.
To tackle it just move the lo and hi by 1 in the forward and backward direction respectively whenever you encounter such condition.

## Code ->
```cpp
class Solution {
public:
    bool search(vector<int>& nums, int target) {
        int lo = 0, hi = nums.size()-1, mid;

        while(lo <= hi){
            mid = (lo+hi)/2;
            if(nums[mid]==target) return true; // if target found, return true

            // the major edge case: if repetation of elements is allowed, our algo will get confused which side to prune
            // eg: [3, 0, 2, 3, 3, 3, 3] here mid elem is 3 and even the hi and lo is 3 so move one step from front and back
            if(nums[mid] == nums[lo] && nums[mid] == nums[hi]){ 
                lo = lo+1;
                hi = hi-1;
                continue; // after the adjustment just continue the logic
            }

            if(nums[mid]>=nums[lo]){ // identified that left side is sorted
                if(nums[mid]>target && nums[lo]<=target) // if the target is between the sorted side
                    hi = mid-1;
                else lo = mid+1; // not between the sorted side
            }
            else{   // identifed that right side is sorted
                if(nums[mid]<target && nums[hi]>=target) // if the target is between the sorted side
                    lo = mid+1;
                else hi = mid-1; // not between the sorted side
            }
        }

        return false;
    }
};
```
# [153. Find Minimum in Rotated Sorted Array](https://leetcode.com/problems/find-minimum-in-rotated-sorted-array/)

## Approach ->
The approach is simple. First identify the sorted half. Pick the smallest element from the sorted half and then update it with answer if it is smaller than answer. Now you can prune the half that you identified as sorted because its job here is done. I named this algorithm the use-and-throw algorithm.

## Code ->
```cpp
class Solution {
public:
    int findMin(vector<int>& nums) {
        int low=0, high=nums.size()-1, mid, ans=INT_MAX;

        while(low<=high){
            mid = low + (high-low)/2;

            // Optimization
            if(nums[low]<=nums[high]){
                ans = min(ans, nums[low]);
                break;
            }

            if(nums[mid]>=nums[low]){ // left half is sorted
                // compare and update the ans with nums[low] and prune the left side
                ans = min(ans, nums[low]); 
                low = mid+1;
            }
            else{
                // compare and update the ans with nums[mid] and prune the right side
                ans = min(ans, nums[mid]);
                high = mid-1;
            }
        }

        return ans;
    }
};
```


# [162. Find Peak Element](https://leetcode.com/problems/find-peak-element/description/)

## Approach ->
First let us understand the question. What is a peak element?

A peak element in an array refers to the element that is greater than both of its neighbors. Basically, if arr[i] is the peak element, arr[i] > arr[i-1] and arr[i] > arr[i+1].

The element outside the bounds is minus infinity so it will always be smaller than the 0th index and last index element.

The approach is fairly simple, since we want to find the element that is greater than its neighbour element so we will be on a persuit of a greater element always. Hence always eliminate the half that has the smaller neighbour element. 
Imagine it to be like a mountain, if we keep climbing to the higher part, we will certainly reach the peak and that peak will be the answer because it will be greater than its left and right elements. And if we are on a persuit of a greater element, it is guaranteed that we fill find a peak (not necessarily the greatest element in the array but a peak means a position which is higher than its left and right positions). We are so sure that we will find a peak because in the question it is clearly stated that outside the bounds of the array there is -1, so no matter what we will find a peak if we keep climbing up the mountain i.e. keep going to the higher element side. Look at the code and you would understand. Remember to handle the edge cases in the beginning and not inside the while loop to avoid long code.

## Code ->
```cpp
class Solution {
public:
    int findPeakElement(vector<int>& nums) {
        int low=1, hi=nums.size()-2, mid, n=nums.size();
        if(n==1) return 0; // if size of nums is 1 return 0
        if(nums[0]>nums[1]) return 0; //if first element is peak return 0
        if(nums[n-1]>nums[n-2]) return n-1; // if last element is peak 

        while(low<=hi){
            mid = (low+hi)/2;

            if(nums[mid]>nums[mid-1] && nums[mid]>nums[mid+1])
                return mid; // if you find peak, return index
            if(nums[mid+1]>nums[mid-1])
                low = mid+1; // if the element to the right of index is greater than the element to the left of index then eliminate the left side completely 
            else hi = mid-1; // else eliminate the right side
        }
        return -1;
    }
};
```

# [540. Single Element in a Sorted Array](https://leetcode.com/problems/single-element-in-a-sorted-array/description/)

## Approach->
You already know the linear search approaches like hashing and bitwise xor but we have to do it in log n time complexity.

Binary search approach:
Let us observe something here and then the code will be a breeze to write. If you observe the indexing here then its something like this -> if an element appears twice in a sequence then its first occurence should be even and second should be odd (normal occurence). For example in `nums = [1,1,2,3,3,4,4,8,8]` -> first 1 has 0 index but second has index as 1 (even, odd). But if the occurences are in reverse i.e. first occurence odd and second even then that means that a single element before the appearance of the two elements has disrupted the index position. For example in above example -> first 3 has index 3 and second 3 has index 4 (odd, even).

Thus we can conclude that if the occurence case is normal (even, odd) for element at mid then we have to eliminate the left half because our single element appears somewhere on the right side. But if the occurence case is (odd, even) then we will have to eliminate the right half.

## Code ->
```cpp
class Solution {
public:
    int singleNonDuplicate(vector<int>& nums) {
         int low = 1, hi = nums.size()-2, mid, n=nums.size();
         if(n==1) return nums[0];
         if(nums[0]!=nums[1]) return nums[0];
         if(nums[n-1]!=nums[n-2]) return nums[n-1]; // check both base conditions at first to avoid long code inside while loop and it gets confusing too if you check base cases with so many if else inside while loop.

         while(low<=hi){
             mid = (low+hi)/2;

             if(nums[mid]!=nums[mid-1] && nums[mid]!=nums[mid+1]) return nums[mid];
             
             // by observation we know that if an element appears twice in a sequence then its first occurence should be even and second should be odd (normal occurence). For example in example 1 -> first 1 has 0 index but second has index as 1. But if the occurences are in reverse i.e. first occurence odd and second even then that means that a single element before the appearance of the two elements has disrupted the index position.

             if(nums[mid]==nums[mid-1]){ // finding if the element before mid is equal to element at mid
                 if(mid%2==1) low = mid+1; // if normal occurence then eliminate left side
                 else hi = mid-1; // eliminate right side
             }
             else{
                 if(mid%2==1) hi = mid-1;
                 else low = mid+1;
             }
         }    

         return -1; 
    }
};
```



# [1011. Capacity To Ship Packages Within D Days](https://leetcode.com/problems/capacity-to-ship-packages-within-d-days/)

## Approaches ->
### _Brute Force:_ 

On observation we can conclude that the possible capacity of the ship must lie between the range of maximum element in the weights array to the summation of all elements in weights array. 

For example -> Input: weights = [1,2,3,4,5,6,7,8,9,10], days = 5
Output: 15

Here the range of possible capacities will lie between 10 to 55.

Algorithm:

-  We will use a loop(say cap) to check all possible capacities.
-  Next, inside the loop, we will send each capacity to the findDays() function to get the number of days required for that particular capacity.
-  The minimum number, for which the number of days <= d, will be the answer.

Time complexity -> O(N * (sum(weights[]) – max(weights[]) + 1)),

### _Optimal Approach using binary search:_

Algorithm:
-  First, we will find the maximum element i.e. max(weights[]), and the summation i.e. sum(weights[]) of the given array.
-  Place the 2 pointers i.e. low and high: Initially, we will place the pointers. The pointer low will point to max(weights[]) and the high will point to sum(weights[]).
-  Calculate the ‘mid’
-  Eliminate the halves based on the number of days required for the capacity ‘mid’:
We will pass the potential capacity, represented by the variable ‘mid’, to the ‘findDays()‘ function. This function will return the number of days required to ship all the weights for the particular capacity, ‘mid’.
-  1. If munerOfDays <= d: On satisfying this condition, we can conclude that the number ‘mid’ is one of our possible answers. But we want the minimum number. So, we will eliminate the right half and consider the left half(i.e. high = mid-1).
-  2. Otherwise, the value mid is smaller than the number we want. This means the numbers greater than ‘mid’ should be considered and the right half of ‘mid’ consists of such numbers. So, we will eliminate the left half and consider the right half(i.e. low = mid+1).
-  Finally, outside the loop, we will return the value of low as the pointer will be pointing to the answer.

## Code ->
```cpp
class Solution {
public:
    int findDays(vector<int> weights, int capacity){
        int days = 0, sum = 0;

        for(int i=0; i<weights.size(); i++){
            sum+=weights[i];
            if(sum<=capacity) continue;
            days++;
            sum=weights[i];
        }
        return ++days;
    }

    int shipWithinDays(vector<int>& weights, int days) {
        // using stl to find high and low.
        int hi = accumulate(weights.begin(), weights.end(), 0);
        int lo = *max_element(weights.begin(), weights.end());
        int mid;

        while(lo<=hi){
            mid = (lo+hi)/2;

            int daysWithMid = findDays(weights, mid);

            // if for mid the amount of days are less than the days given then that probably means that mid is way more than it should be, that's why it was able to accomodate all weights in less number of days, hence eliminate the right half and keep searching on the left.
            if(daysWithMid<=days) hi = mid-1; 
            else lo = mid+1;
        }
        return lo; // lo will keep pointing on the most optimal mid always
    }
};
```

# [ Number of occurrence](https://www.codingninjas.com/studio/problems/occurrence-of-x-in-a-sorted-array_630456?utm_source=striver&utm_medium=website&utm_campaign=a_zcoursetuf&leftPanelTabValue=PROBLEM)

## Approach ->
(Last occurence - First occurence + 1)
and if there is no occurence then return 0

## Code ->
```cpp
int firstOccurrence(vector<int> &arr, int n, int k) {
    int low = 0, high = n - 1;
    int first = -1;

    while (low <= high) {
        int mid = (low + high) / 2;
        // maybe an answer
        if (arr[mid] == k) {
            first = mid;
            //look for smaller index on the left
            high = mid - 1;
        }
        else if (arr[mid] < k) {
            low = mid + 1; // look on the right
        }
        else {
            high = mid - 1; // look on the left
        }
    }
    return first;
}

int lastOccurrence(vector<int> &arr, int n, int k) {
    int low = 0, high = n - 1;
    int last = -1;

    while (low <= high) {
        int mid = (low + high) / 2;
        // maybe an answer
        if (arr[mid] == k) {
            last = mid;
            //look for larger index on the right
            low = mid + 1;
        }
        else if (arr[mid] < k) {
            low = mid + 1; // look on the right
        }
        else {
            high = mid - 1; // look on the left
        }
    }
    return last;
}

int count(vector<int>& arr, int n, int x) {
	int l = lastOccurrence(arr, n, x);
	int f = firstOccurrence(arr, n, x);
	if(f==-1) return 0; // if no occurence return 0
	return l-f+1;
}
```

# [69. Sqrt(x)](https://leetcode.com/problems/sqrtx/description/)

## Approach ->
The sqrt of a number will always lie in the range of 1 to the number itself. So we will check for mid*mid and if it is smaller than or equal to n then it might be the answer, so eleminate the left half. You can store the ans in ans variable or simply return high at end because if you think about it high will always point to the answer.

## Code ->
```cpp
class Solution {
public:
    int mySqrt(int x) {
        // using binary search
        int low = 1, high = x;
        //Binary search on the answers:
        while (low <= high) {
            // always calculate mid using this formula to avoid overflow
            long long mid = low + (high-low)/2;
            long long val = mid * mid;
            if (val <= (long long)(x)) {
                //eliminate the left half:
                low = mid + 1;
            }
            else {
                //eliminate the right half:
                high = mid - 1;
            }
        }
        return high;
    }
};
```
# [875. Koko Eating Bananas](https://leetcode.com/problems/koko-eating-bananas/description/)

## Approaches ->
1. The extremely naive approach is to check all possible answers from 1 to max(a[]). The minimum number for which the required time <= h, is our answer.
2. Apply binary search 

## Code ->
```cpp
class Solution {
public:
    int calculateTotalHours(vector<int> &piles, int hourly) {
        long long totalH = 0;
        int n = piles.size();
        //find total hours
        // Don't use loop to find the hours, instead you can use division in O(1) tc.
        for (int i = 0; i < n; i++) {
            totalH += ceil((double)(piles[i]) / (double)(hourly));
        }
        return totalH;
    }

    int minEatingSpeed(vector<int>& piles, int h) {
        int low = 1, high = *max_element(piles.begin(), piles.end());

        while (low <= high) {
            int mid = (low + high) / 2;
            int totalH = calculateTotalHours(piles, mid);
            // For the given number of bananas to be eaten per hour (mid), total hour taken to eat complete pile is calculated. If this total hour is smaller than the given hour then that means koko is eating too fast, so we will lower the value of mid(no. of bananas per hour)
            if (totalH <= h) {
                high = mid - 1;
            }
            // else just make the value of mid more
            else {
                low = mid + 1;
            }
        }
        return low;
        // low will point the required answer because as we know the rule of returning i.e. if there is a possible answer, and we eleminate a side, we return the opposite to the eliminated side. We can store the ans in ans variable as well if confusing.
    }
};
```

Potential mistakes while coding this approach:
1. Returning mid instead of low
2. Initializing low as 0 instead of 1. If 0 is initialized then we will get a runtime error because division by zero is undefined.
3. Using a loop to find totalHour instead of division.

TC-> O(n⋅log(max_pile))
​SC-> O(1)


# [1482. Minimum Number of Days to Make m Bouquets](https://leetcode.com/problems/minimum-number-of-days-to-make-m-bouquets/description/)

## Approach->
The solution is very simple when you understand the question properly. 

Input Format: N = 8, arr[] = {7, 7, 7, 7, 13, 11, 12, 7}, m = 2, k = 3
Result: 12

Let's grasp the question better with the help of an example. Consider an array: {7, 7, 7, 7, 13, 11, 12, 7}. We aim to create bouquets with k, which is 3 adjacent flowers, and we need to make m, which is 2 such bouquets. Now, if we try to make bouquets on the 11th day, the first 4 flowers and the 6th and the last flowers would have bloomed. So, we will be having 6 flowers in total on the 11th day. However, we require two groups of 3 adjacent flowers each. Although we can form one group with the first 3 adjacent flowers, we cannot create a second group. Therefore, 11 is not the answer in this case.

So basically what we need to do is, for a given day we will keep counting the number of adjacent blooming flower and when that number is equal to k then that means that one bouqet can be made. By doing this we can find the total number of bouquets made for a given day and compare it with m to find our answer. Use binary search to do it in the most optimal way

## Code ->
```cpp
class Solution {
public:
    // returns the number of booq that can be made for mid's value (day) 
    int numberOfBouq(vector<int> bloomDay, int k, int days){
        int numBook = 0, numAdj=0;
        for(int i=0; i<bloomDay.size(); i++){
            // if the flower is blooming, increase the value of numAdj else make numAdj 0
            if(days>=bloomDay[i]) numAdj++;
            else numAdj = 0;


            // if numAdj is equal to k then that means one bouquet can be made. Reset numAdj.
            if(numAdj==k){
                numBook++;
                numAdj=0;
            }
        }

        return numBook;
    }
    int minDays(vector<int>& bloomDay, int m, int k) {
        int maxNum = *max_element(bloomDay.begin(), bloomDay.end());
        int low = 0, high = maxNum, mid, ans=-1;

        while(low<=high){
            mid = low + (high-low)/2;
            int numBouq = numberOfBouq(bloomDay, k, mid);

            // if number of bouq for a given day is greater than equal to required bouq then that day can be the potential answer.
            if(numBouq>=m){
                high = mid-1;
                ans = mid;
            }
            else low = mid+1;
        }
        return ans;
    }
};
```
# [1283. Find the Smallest Divisor Given a Threshold](https://leetcode.com/problems/find-the-smallest-divisor-given-a-threshold/description/)

## Code ->
```cpp
class Solution {
public:
    // Fn. to find the summation of division values
    int sumByD(vector<int> &arr, int div) {
        int n = arr.size();
        int sum = 0;
        for (int i = 0; i < n; i++) {
            sum += ceil((double)(arr[i]) / (double)(div));
        }
        return sum;
    }
    int smallestDivisor(vector<int>& nums, int threshold) {
        // lower the divisor, higher will be the sum by divisor, so we will try to find the lowest divisor that satisfies the conditions
        int n = nums.size();

        // note: initialized low with 1 instead of 0.
        int low = 1, high = *max_element(nums.begin(), nums.end());

        //Apply binary search:
        while (low <= high) {
            int mid = (low + high) / 2;
            if (sumByD(nums, mid) <= threshold) {
                // this is a possible answer but let's try reducing the divisor
                high = mid - 1;
            }
            else {
                low = mid + 1;
            }
        }
        return low;
    }
};
```

# [1539. Kth Missing Positive Number](https://leetcode.com/problems/kth-missing-positive-number/description/)

## [Approaches](https://takeuforward.org/arrays/kth-missing-positive-number/)

## Codes
1. Brute Force:
```cpp
class Solution {
public:
    int findKthPositive(vector<int>& arr, int k) {
        for(int i=0; i<arr.size(); i++){
            if(k<arr[i]) return k;
            k++;
        }
        return k;
    }
};
```

2. Binary Search
```cpp
class Solution {
public:
    // Function to find the kth missing positive integer in a strictly increasing array
    int findKthPositive(vector<int>& arr, int k) {
        // Get the size of the input array
        int n = arr.size();

        // Initialize low and high pointers for binary search
        int low = 0, high = n - 1;

        // Binary search to find the position where the kth missing positive integer lies
        while (low <= high) {
            // Calculate the middle index
            int mid = (low + high) / 2;

            // Calculate the number of missing positive integers before the middle element
            int missing = arr[mid] - (mid + 1);

            // If the number of missing positive integers before mid is less than k
            if (missing < k) {
                // Update low to search in the right half of the array
                low = mid + 1;
            }
            else {
                // Update high to search in the left half of the array
                high = mid - 1;
            }
        }

        // Return the kth missing positive integer
        // k + high + 1 gives the actual missing positive integer in the array
        return k + high + 1;
    }
};
```

### Max-min/min-max problems start here

# [ Aggressive Cows](https://www.codingninjas.com/studio/problems/aggressive-cows_1082559?utm_source=striver&utm_medium=website&utm_campaign=a_zcoursetuf)

### Min-max problem

## Intuition->

You are given **n** stall positions and need to place **k cows** such that the **minimum distance between any two cows** is **maximized**.

This is a classic **Binary Search on Answer** problem, where the idea is:

* **We want to maximize the minimum distance** between any two cows.
* So, we search in a range of distances: `low = 1` (smallest possible distance), `high = max(stalls) - min(stalls)` (largest possible distance).
* For a given distance `mid`, we check: **is it possible to place all cows such that no two cows are closer than `mid`?**

  * If yes → we try to increase the distance.
  * If no → we reduce the distance.

---

### Why does this work?

This is essentially trying to find the **largest distance** `d` such that we can place all `k` cows with at least `d` distance apart.

This is a **monotonic decision function**:

* If placing cows is possible at distance `d`, then it's also possible for any distance `< d`.
* If it's **not possible** at distance `d`, then it's definitely **not possible** at any `d > current`.

So, we can use binary search to efficiently find the answer.

---

### Why do we return `high`?

This is the core observation:

* In Binary Search on Answer problems, we search for a maximum (or minimum) **feasible** value.
* During the loop:

  * If `mid` is **feasible** (we can place cows), we go **right** (`low = mid + 1`) to try for a larger value.
  * If `mid` is **not feasible**, we go **left** (`high = mid - 1`) to try a smaller value.

Eventually:

* `low` will move **past** the last feasible value (to the impossible side),
* `high` will be at the **last feasible value** — the maximum possible minimum distance.

Thus, **`high` is our answer**, and no need to store it in a separate variable.

---


## Code ->
```cpp
// Helper function to count how many cows can be placed
// such that the minimum distance between them is at least 'mid'
int findNumOfCows(vector<int> &stalls, int mid) {
    int cows = 1; // First cow placed at the first stall
    int lastPos = stalls[0]; // Position of the last placed cow

    // Iterate through stalls to place cows
    for (int i = 1; i < stalls.size(); i++) {
        // If the distance from last placed cow is at least 'mid', place a new cow
        if (stalls[i] - lastPos >= mid) {
            cows++;
            lastPos = stalls[i]; // Update last placed cow position
        }
    }

    return cows; // Return total number of cows placed
}

// Main function to find the maximum minimum distance
int aggressiveCows(vector<int> &stalls, int k) {
    sort(stalls.begin(), stalls.end()); // Sort the stall positions

    int low = 1; // Minimum possible distance between cows
    int high = stalls.back() - stalls.front(); // Maximum possible distance
    int mid;

    // Binary search to find the largest minimum distance possible
    while (low <= high) {
        mid = low + (high - low) / 2; // Mid distance to test

        int numOfCows = findNumOfCows(stalls, mid); // Try placing cows with 'mid' distance

        if (numOfCows >= k) {
            // If we can place at least 'k' cows, try for a bigger distance
            low = mid + 1;
        } else {
            // Otherwise, reduce the distance
            high = mid - 1;
        }
    }

    // 'high' now points to the maximum minimum distance possible
    return high;
}

```

# [ Allocate Books](https://www.codingninjas.com/studio/problems/allocate-books_1090540?utm_source=youtube&utm_medium=affiliate&utm_campaign=codestudio_Striver_BinarySeries&leftPanelTabValue=PROBLEM)
### Max-min problem. 

## Intuition
The problem requires allocating books to students in a contiguous manner such that the maximum number of pages assigned to any student is minimized. This is a classic optimization problem that can be efficiently solved using binary search. The key idea is to determine the minimum possible maximum pages any student has to read by checking feasible allocations within a search space defined by the maximum pages in a single book (lower bound) and the total sum of all pages (upper bound).

## Approach
1. **Binary Search Setup**: 
   - **Lower Bound (`low`)**: The maximum number of pages in a single book. This ensures each student gets at least one book.
   - **Upper Bound (`high`)**: The sum of all pages, representing the scenario where one student reads all books.

2. **Binary Search Execution**:
   - For each midpoint (`mid`) in the current search range, calculate the number of students required to allocate all books such that no student reads more than `mid` pages.
   - **Feasibility Check**: If the number of students required (`numOfStu`) is less than or equal to `m`, it means `mid` is a feasible solution. We then try to find a smaller `mid` by adjusting the upper bound (`high = mid - 1`).
   - If `numOfStu` exceeds `m`, it means `mid` is too small, so we adjust the lower bound (`low = mid + 1`) to search for a larger `mid`.

3. **Termination**: The loop terminates when `low` exceeds `high`, at which point `low` will point to the minimum possible maximum pages that can be allocated to `m` students.

## Code->
```cpp
// Function to calculate the number of students needed if each student can read at most 'pages' pages
int findNumOfStu(vector<int> &arr, int pages) {
    int students = 1;  // At least one student is needed
    int currentPages = 0;  // Pages allocated to the current student

    for (int i = 0; i < arr.size(); i++) {
        // If adding the current book exceeds the 'pages' limit, allocate to a new student
        if (currentPages + arr[i] > pages) {
            students++;
            currentPages = arr[i];  // Start new allocation for the new student
        } else {
            currentPages += arr[i];  // Add the current book to the current student's allocation
        }
    }
    return students;
}

// Function to find the minimum possible maximum pages allocated to any student
int findPages(vector<int>& arr, int n, int m) {
    if (m > n) return -1;  // More students than books, allocation not possible

    int low = arr[0];  // Initialize with the first book's pages
    int high = 0;      // Will hold the total sum of all pages

    // Determine the lower and upper bounds for binary search
    for (int i = 0; i < n; i++) {
        low = max(low, arr[i]);  // 'low' is the maximum pages in any single book
        high += arr[i];          // 'high' is the sum of all pages
    }

    // Binary search to find the minimum possible maximum pages
    while (low <= high) {
        int mid = low + (high - low) / 2;  // Midpoint to check

        int numOfStu = findNumOfStu(arr, mid);  // Number of students needed for 'mid' pages

        if (numOfStu <= m) {
            // If feasible, try to find a smaller 'mid' by reducing the upper bound
            high = mid - 1;
        } else {
            // If not feasible, increase the lower bound to find a larger 'mid'
            low = mid + 1;
        }
    }

    // 'low' holds the minimum possible maximum pages after the loop terminates
    return low;
}
```


### watch vid if you still don't understand

# [410. Split Array Largest Sum](https://leetcode.com/problems/split-array-largest-sum/description/)

## [Approaches](https://takeuforward.org/arrays/split-array-largest-sum/)

## Codes ->
1. Brute Force
```cpp
class Solution {
public:
    int numOfSubarrays(vector<int> &nums, int check){
        int sub = 1, sum = 0;

        for(int i=0; i<nums.size(); i++){
            if(sum + nums[i] > check){
                sub++;
                sum = nums[i];
            }
            else{
                sum += nums[i];
            }
        }
        return sub;
    }

    int splitArray(vector<int>& nums, int k) {
        int low = *max_element(nums.begin(), nums.end());
        int high = accumulate(nums.begin(), nums.end(), 0);
        int mid;

        for(int i=low; i<=high; i++){
            int sub = numOfSubarrays(nums, i);
            cout << sub << " " << i << endl;
            if(sub == k) return i;
        }
        return low; 
    }
};
```
2. BS
```cpp
class Solution {
public:
    int numOfSubarrays(vector<int> &nums, int check){
        int sub = 1, sum = 0;

        for(int i=0; i<nums.size(); i++){
            if(sum + nums[i] > check){
                sub++;
                sum = nums[i];
            }
            else{
                sum += nums[i];
            }
        }
        return sub;
    }

    int splitArray(vector<int>& nums, int k) {
        int low = *max_element(nums.begin(), nums.end());
        int high = accumulate(nums.begin(), nums.end(), 0);
        int mid;

        while(low <= high){
            mid = low + (high-low)/2;

            int sub = numOfSubarrays(nums, mid);

            // note that the condition over here is sub>k and not sub>=k beacuse we are trying to find min
            if(sub>k) low = mid+1; 
            else high = mid - 1;
        }
        // note that we are returning low and not high. 
        return low;
    }
};
```
# [Painter's Partition Problem](https://www.codingninjas.com/studio/problems/painter-s-partition-problem_1089557?utm_source=striver&utm_medium=website&utm_campaign=a_zcoursetuf)

## Approach 
Same as of last two questions

# [Row with max 1s](https://www.codingninjas.com/studio/problems/row-of-a-matrix-with-maximum-ones_982768?utm_source=striver&utm_medium=website&utm_campaign=a_zcoursetuf)

## Code ->
```cpp
// Function to count the number of 1s in a sorted binary row using binary search
int findNumOfOnes(vector<int> matrix) {
    int low = 0, mid, high = matrix.size();

    while (low <= high) {
        mid = low + (high - low) / 2;

        if (matrix[mid] == 0)
            low = mid + 1;  // Move to the right half if mid is 0
        else
            high = mid - 1; // Move to the left half if mid is 1
    }

    // The number of 1s is the total elements minus the index of the first 1
    return matrix.size() - low;
}

// Function to find the row with the maximum number of 1s in a binary matrix
int rowWithMax1s(vector<vector<int>> &matrix, int n, int m) {
    int row = -1;       // Initialize row index to -1 (no row found)
    int ones = 0;       // Initialize max count of 1s to 0

    // Iterate through each row to find the one with the most 1s
    for (int i = 0; i < n; i++) {
        int numOfOnes = findNumOfOnes(matrix[i]);

        // Update row index if current row has more 1s than the previous max
        if (numOfOnes > ones) {
            row = i;
            ones = numOfOnes;
        }
    }

    return row;
}
```

# [74. Search a 2D Matrix](https://leetcode.com/problems/search-a-2d-matrix/description/)

## Approaches ->
1. Brute Force - Traverse in O(n*m) TC and find the target.
2. Better Approach: Place yourself on either the top right or bottom left cornor, that way when you go to your left the elements decrease and when you go down the elements increase. Using this you can find the target in O(n+m) TC.
3. Most Optimal: Treat 2D array as 1D array and perform Binary Search. TC -> O(log(n*m))

## Codes->
2.
```cpp
class Solution {
public:
    bool searchMatrix(vector<vector<int>>& matrix, int target) {
        int row = 0, col = matrix[0].size()-1;

        while(row<matrix.size() && col>=0){
            if(matrix[row][col]==target) return true;
            else if(matrix[row][col]>target) col--;
            else row++;
        }
        return false;
    }
};
```
3. 
```cpp
class Solution {
public:
    bool searchMatrix(vector<vector<int>>& matrix, int target) {
        int m = matrix[0].size(); int n = matrix.size();

        int low = 0, high = (m*n)-1, mid;

        while(low<=high){
            mid = (low+high)/2;

            // Treat 2d as 1d using this formula
            int val = matrix[mid/m][mid%m];

            if(val==target) return true;
            if(val>target) high = mid-1;
            else low = mid+1;
        }
        return false;
    }
};
```

# [240. Search a 2D Matrix II](https://leetcode.com/problems/search-a-2d-matrix-ii/description/)

## Approach ->
The approach 2 of the above question works here.

# [1901. Find a Peak Element II](https://leetcode.com/problems/find-a-peak-element-ii/description/)

## Approach ->
A little tweak in find peak element 1, go revise the hill concept of that q and then come here.
The approach involves mimicking the experience of traversing a mountain range, column by column. For each column, we identify the highest point. Then, we see the immediate left and right from this peak to determine if it's the peak element (no need to check up and down because it is the greatest element we've picked in the column). If a higher point exists to the right, we eliminate the left half of the search space; if it's on the left, we eliminate the right half. This process continues until we find the peak or exhaust the search space. 

## Code ->
```cpp
class Solution {
public:
    // Helper function to find the row with the greatest value in a specific column
    int findGreatest(vector<vector<int>> &mat, int col) {
        int maxi = mat[0][col], row = 0;
        for (int i = 1; i < mat.size(); i++) {
            if (mat[i][col] > maxi) {
                maxi = mat[i][col];
                row = i;
            }
        }
        return row;
    }

    // Main function to find a peak element in a 2D grid
    vector<int> findPeakGrid(vector<vector<int>> &mat) {
        int row = mat.size() - 1, col = mat[0].size() - 1;
        int low = 0, high = col, mid;

        // Binary search for the peak element
        while (low <= high) {
            mid = low + (high - low) / 2;

            // Find the row with the greatest value in the current column
            row = findGreatest(mat, mid);

            // Determine values to the left and right of the current position
            int left = (mid - 1 >= 0) ? mat[row][mid - 1] : -1;
            int right = (mid + 1 <= col) ? mat[row][mid + 1] : -1;
            int val = mat[row][mid];

            // Check if the current position is a peak element
            if (val > left && val > right)
                return {row, mid};
            else if (val < left)
                high = mid - 1;
            else
                low = mid + 1;
        }
        // If no peak element is found, return {-1, -1}
        return {-1, -1};
    }
};
```