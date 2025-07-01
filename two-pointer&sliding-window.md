# [3. Longest Substring Without Repeating Characters](https://leetcode.com/problems/longest-substring-without-repeating-characters/description/)

## Approaches ->
You know the brute force, let's look at the optimized sol. We will have two pointers left and right. Pointer ‘left’ is used for maintaining the starting point of the substring while ‘right’ will maintain the endpoint of the substring.’ right’ pointer will move forward and check for the duplicate occurrence of the current element if found then the ‘left’ pointer will be shifted ahead so as to delete the duplicate elements. We will make a map that will take care of counting the elements and maintaining the frequency of each and every element as a unity by taking the latest index of every element.

## Code ->
```cpp
class Solution {
public:
    int lengthOfLongestSubstring(string s) {
        // Check if the string is empty, return 0 if true
        if(s.size() == 0) 
            return 0;

        // Create a map to store the latest index of each character in the string
        unordered_map<int, int> mp;

        // Initialize two pointers 'l' and 'r', and 'ans' to store the final result
        int l = 0, r = 0, ans = 0;

        // Iterate through the string using the 'r' pointer
        while(r < s.size()) {
            // Check if the current character at 'r' is already in the map and its index is greater than or equal to 'l'
            if(mp.find(s[r]) != mp.end() && mp[s[r]] >= l) {
                // If true, move 'l' pointer to the next position after the last occurrence of the current character
                l = mp[s[r]] + 1;
            }

            // Update the maximum length of substring without repeating characters
            ans = max(ans, r - l + 1);

            // Update the index of the current character in the map
            mp[s[r]] = r;

            // Move 'r' pointer to the next character
            r++;
        }

        // Return the final result
        return ans;
    }
};
```

# [1004. Max Consecutive Ones III](https://leetcode.com/problems/max-consecutive-ones-iii/description/)

## Code ->
```cpp
class Solution {
public:
    int longestOnes(vector<int>& nums, int k) {
        int i = 0, j = 0, ans = 0, temp = 0, zeroCnt = 0;

        // Iterate through the array using two pointers (i and j)
        while (j < nums.size()) {
            // Check if the current element at the right pointer (j) is 0, cause if it is 1 you just keep moving j and keep updating answer 
            if (nums[j] == 0) {
                zeroCnt++; // Increment the count of zeros
                // If the count of zeros exceeds the allowed flips (k),
                // move the left pointer (i) until you eliminate a 0 from the window
                if (zeroCnt > k) {
                    while (i <= j) {
                        if (nums[i] == 0) {
                            zeroCnt--; // Decrement the count as a zero eliminated from the window
                            i++; // Move the left pointer to the right
                            break; // Exit the loop once a zero is flipped back
                        }
                        i++; // Move the left pointer to the right
                    }
                }
            }
            j++; // Move the right pointer to the right
            ans = max(ans, j - i); // Update the maximum length of subarray with at most k flips
        }
        return ans; // Return the final result
    }
};
```

# [904. Fruit Into Baskets](https://leetcode.com/problems/fruit-into-baskets/description/)

## Approach ->
Maintain a window from l to r where you're moving. Maintain a map to keep track of the types of fruits in the two baskets as you move, if more than 2 types of fruits come in the basket then remove from the left till one type is completly gone. Then keep calculating your answer as you move.
## Code ->
```cpp
class Solution {
public:
    int totalFruit(vector<int>& fruits) {
        // Initialize pointers 'l' and 'r', and variable 'ans' to store the maximum number of fruits
        int l = 0, r = 0, ans = 0;

        // Create an unordered_map to store the count of each fruit type in the current window
        unordered_map<int, int> mp;

        // Iterate through the array using the 'r' pointer
        while (r < fruits.size()) {
            // Update the count of the current fruit in the map
            mp[fruits[r]]++;

            // Adjust 'l' pointer if more than two types of fruits are present in the window
            while (mp.size() > 2) {
                // keep subtracting the frequency of fruits on the left and keep moving the left at the end
                mp[fruits[l]]--;

                // If the count of a fruit becomes zero, that means our baskets have only one type of fruit left
                // so remove the history of that fruit from the map
                if (mp[fruits[l]] == 0)
                    mp.erase(fruits[l]);

                // keep moving the left pointer
                l++;
            }

            // Update the maximum number of fruits picked in the current window
            ans = max(ans, r - l + 1);

            // Move 'r' pointer to the next position
            r++;
        }

        // Return the final result
        return ans;
    }
};
```
# [424. Longest Repeating Character Replacement](https://leetcode.com/problems/longest-repeating-character-replacement/description/)

## Approach ->
The approach is simple.. we will keep updating the window and its elements in a map. If the windowSize - maxRepeating (windowsize-maxrepeating gives us the value of extra elements in the window that we need to convert to make the window largest) element comes out to be greater than k then that means that we have other elements in the map that exceed the value of k. So we remove that element's frequency from the map and move our window.

GPT Explanation:

The approach involves maintaining a sliding window and updating the frequency of each character within the window using a map. The key insight is to track the difference between the window size and the maximum frequency of any character in the window (windowSize - maxRepeating). This difference represents the number of extra elements in the window that need to be converted to make the window as large as possible.

If this difference becomes greater than the allowed operations (k), it implies there are other characters in the window exceeding the permitted modifications. To address this, the algorithm adjusts the window by decrementing the frequency of the leftmost character, effectively moving the left boundary of the window. This process continues iteratively, and at each step, the length of the current window is compared with the maximum length encountered so far, ensuring the result reflects the longest substring containing the same letter achievable within the given constraints.

## Code ->
```cpp
class Solution {
public:
    int characterReplacement(string s, int k) {
        // Create a map to store the frequency of each character in the window
        unordered_map<char, int> mp;
        
        // Pointers for the left and right boundaries of the window
        int l = 0, r = 0;
        
        // Variable to store the maximum repeating character count in the current window
        int maxRepeating = 0;
        
        // Variable to store the length of the longest substring
        int ans = 0;

        // Iterate through the string
        while (r < s.size()) {
            // Update the frequency of the current character in the window
            mp[s[r]]++;
            
            // Update the maximum repeating character count in the current window
            maxRepeating = max(maxRepeating, mp[s[r]]);
            
            // Calculate the size of the current window
            int windowSize = r - l + 1;

            // If the number of elements to be changed to make all elements the same
            // is greater than k, then we need to shrink the window
            if (windowSize - maxRepeating > k) {
                // Reduce the frequency of the leftmost character in the window
                mp[s[l]]--;
                
                // Move the left boundary of the window to the right
                l++;
            } else {
                // Update the length of the longest substring
                ans = max(ans, windowSize);
            }

            // Move the right boundary of the window to the right
            r++;
        }

        // Return the length of the longest substring
        return ans;
    }
};
```

# [930. Binary Subarrays With Sum](https://leetcode.com/problems/binary-subarrays-with-sum/description/)

## Approaches ->
- Brute Force
- Exactly same as subarray sum equals k -> Maintain the prefix sum of the elements of the array (don't make a separate vector for it, keep it in an integer). Make a hashmap. If the prefix sum is equal to k then increase the answer with 1 as because you were directly able to find the sum of subarray equal to k. Now find if the value of prefix sum - k exist in the map. If it does then add the frequency of sum - k to answer. Now update the map (insert the subarray sum in the map).
- Most optimal method: Using 2 pointer + atmost method, you will find these approaches later on in the same .md file, just keep going.

## Code ->
2. 
```cpp
class Solution {
public:
    int numSubarraysWithSum(vector<int>& nums, int goal) {
        unordered_map<int, int> mp;
        int sum = 0, ans = 0;

        for(int i=0; i<nums.size(); i++){
            sum += nums[i];
            if(sum==goal) ans++;
            if(mp.count(sum-goal)) ans+=mp[sum-goal];

            mp[sum]++;
        }   
        return ans;

    }
};
```

# [1248. Count Number of Nice Subarrays](https://leetcode.com/problems/count-number-of-nice-subarrays/description/)

## Approaches ->
Same as last approaches and the atmost method you'll find later in the same .md file just keep going.

## Code ->
```cpp
class Solution {
public:
    int numberOfSubarrays(vector<int>& nums, int k) {
        unordered_map<int, int> mp;
        int sum = 0, ans = 0;

        for(int i=0; i<nums.size(); i++){
            if(nums[i]%2==1) sum++;

            if(sum==k) ans++;
            if(mp.count(sum-k)) ans += mp[sum-k];

            mp[sum]++;
        }
        return ans;
    }
};
```

# [1358. Number of Substrings Containing All Three Characters](https://leetcode.com/problems/number-of-substrings-containing-all-three-characters/description/)

## Approach ->
Read the code first and if you don't understand then read the approach. 

The approach involves maintaining a sliding window with pointers 'l' and 'r' to iterate through the string 's'. An unordered map 'mp' is used to track the frequency of characters within the current window. The algorithm increments the frequency of the character at the 'r' pointer.

Within the loop, another while loop checks if all three characters 'a', 'b', and 'c' are present in the current window. If so, it implies that substrings containing at least one occurrence of each character can be formed. The number of such substrings is then added to the result 'ans', where the count is determined by the remaining length of the string 's' from the 'r' pointer.

To maintain the validity of the window, the left boundary 'l' is incrementally moved, and the frequency of the character at the 'l' pointer is decremented until at least one of the characters is no longer present in the window.

This process continues until the 'r' pointer reaches the end of the string 's'. The final result 'ans' represents the total number of substrings containing at least one occurrence of all three characters 'a', 'b', and 'c'.

## Code ->
```cpp
class Solution {
public:
    int numberOfSubstrings(string s) {
        int l = 0, r = 0, ans = 0, n = s.size();
        unordered_map<char, int> mp;

        while(r<n){
            // Update the frequency of the character at the 'r' pointer
            mp[s[r]]++;

            // Check if all three characters 'a', 'b', and 'c' are present in the current window
            while(mp['a'] && mp['b'] && mp['c']){
                // no. of substr that contains a, b, c will not just be the substr that's in the current window but also all the substrings after the current window as well i.e. from last pos to r will also contain a, b, c and they are substrings as well. 
                // eg-> in "abcabc" if 'r' is at index 2 and 'l' at index 0 then the current window i.e. "abc" is a substring containing a, b and c but the substrings after than like "abca", "abcab", "abcabc" also contain a, b, and c. Hence we add s.size()-r to our answer
                ans += n-r; 

                // Move the left boundary 'l' to maintain a valid window
                mp[s[l]]--;
                l++;
            }
            // Move the right boundary 'r' to expand the window
            r++;
        }
        return ans;
    }
};
```


# [1423. Maximum Points You Can Obtain from Cards](https://leetcode.com/problems/maximum-points-you-can-obtain-from-cards/)

## Approach ->
Sliding window Technique -> The approach is fairly simple. We have to pick the cards from either the front or the back. We can also pick the cards from both front and back. So in order to find the maximum score we can find the minimum score to make the question simpler. First, we take a window whose size is total size - k. Then we find the least sum in that window. And finally we subtract the total sum of the array with the least sum to find the maximum sum. Note-> We need not calculate the total sum, that will be covered in prefix sum itself.

## Code ->
1. Using prefixSum
```cpp
class Solution {
public:
    int maxScore(vector<int>& cardPoints, int k) {
        vector<int> prefixSum(cardPoints.size()+1, 0);
   
        //finding the prefix sum
        for(int i=0; i<cardPoints.size(); i++)
            prefixSum[i+1] = prefixSum[i] + cardPoints[i];
       
        //declaring the minimum sum and window size
        int minSum = INT_MAX;
        int window = cardPoints.size()-k; // window will be total size - k
        int i = 1;
       
        //run the loop for the given window and update the minimum sum.
        while(window<prefixSum.size()){
            int sum = prefixSum[window] - prefixSum[i-1];
            minSum = min(minSum, sum);
            window++; i++;
        }

        //return total sum - minimum sum
        return prefixSum[prefixSum.size()-1] - minSum;
    }
};
```
2. Using simple variables 
```cpp
class Solution {
public:
    int maxScore(vector<int>& cardPoints, int k) {
        // Calculate the total sum of all cards
        int totalSum = accumulate(cardPoints.begin(), cardPoints.end(), 0);

        // If the number of cards to be selected is equal to the total number of cards,
        // return the total sum, as we can select all cards.
        if (k == cardPoints.size()) 
            return totalSum;

        // Initialize variables for the sliding window approach
        int ans = INT_MAX, l = 0, r = 0, sum = 0;

        // Update k to represent the size of the desired subarray
        k = cardPoints.size() - k;

        // Sliding window approach
        while (r < cardPoints.size()) {
            sum += cardPoints[r];

            // Check if the size of the current window is equal to k
            if (r - l + 1 == k)
                ans = min(ans, sum);

            r++;

            // If the window size exceeds k, move the left pointer to maintain the window size
            if (r - l + 1 > k) {
                sum -= cardPoints[l];
                l++;
            } 
        }

        // Return the maximum score, which is the total sum minus the minimum sum within the window
        return totalSum - ans;
    }
};
```

# [Longest Substring with At Most K Distinct Characters](https://www.codingninjas.com/studio/problems/longest-substring-with-at-most-k-distinct-characters_2221410?utm_source=striver&utm_medium=website&utm_campaign=a_zcoursetuf&leftPanelTabValue=PROBLEM)

## Code ->
Easy peasy
```cpp
int kDistinctChars(int k, string &str)
{
    int l = 0, r = 0, ans = 0;
    unordered_map<char, int> mp;

    while(r<str.size()){
        mp[str[r]]++;
        while(mp.size()>k){
            mp[str[l]]--;
            if(mp[str[l]]==0) mp.erase(str[l]);
            l++;
        }
        ans = max(ans, r-l+1);
        r++;
    }
    return ans;
}
```

# [992. Subarrays with K Different Integers](https://leetcode.com/problems/subarrays-with-k-different-integers/description/)

## Approach ->
The intuition behind the code lies in the observation that to find the number of subarrays with exactly k distinct elements, we can utilize the concept of "atmost" distinct elements. 

By subtracting the count of subarrays with at most (k-1) distinct elements from the count of subarrays with at most k distinct elements, we obtain the desired result.
In other words -> if we take example of k being 3, if we want to find the substrings with exactly 3 distint elements then we can apply the formula -> elements with atmost 3 distinct elements - elements with atmost 2 distinct elements == elements with exactly 3 distinct elements

 This is because the difference represents the number of subarrays that contain exactly k distinct elements. In simpler terms, for a given k, the formula helps calculate the unique subarrays by considering those with at most k distinct elements and excluding those with at most (k-1) distinct elements.

## Code ->
```cpp
class Solution {
public:
    // Helper function to count subarrays with at most k distinct elements
    int atmost(vector<int> &nums, int k){
        int l = 0, r = 0, count = 0;
        unordered_map<int, int> mp;

        while(r < nums.size()){
            mp[nums[r]]++;

            // Shrink the window until there are at most k distinct elements
            while(mp.size() > k){
                mp[nums[l]]--;
                if(mp[nums[l]] == 0) 
                    mp.erase(nums[l]);
                l++;
            }
            
            // Count subarrays within the current window and include them in the result
            count += r - l + 1;
            r++;
        }
        
        return count;
    }

    int subarraysWithKDistinct(vector<int>& nums, int k) {
        // Apply the formula: elements with at most k distinct elements - elements with at most (k-1) distinct elements
        return atmost(nums, k) - atmost(nums, k-1);
    }
};
```

### Now let us look at some of the questions we did earlier than can be solved using the atmost logic...

# [930. Binary Subarrays With Sum](https://leetcode.com/problems/binary-subarrays-with-sum/description/)

## Approach ->
Using atmost method we can find out the number of subarrays without using any extra space complexity and with an SC of O(1) unlike the map method of this question. Please solve this question using both methods for proper understanding.

## Code ->
```cpp
class Solution {
public:
    int atmost(vector<int> &nums, int goal){
        int l = 0, r = 0, numOfSubarr = 0, count = 0;
        while(r<nums.size()){
            count += nums[r];

            while(count>goal && l<=r){
                count -= nums[l];
                l++;
            }

            numOfSubarr += r-l+1; 
            r++;
        }
        return numOfSubarr;
    }
    int numSubarraysWithSum(vector<int>& nums, int goal) {
        return atmost(nums, goal) - atmost(nums, goal-1);
    }
};
```

# [1248. Count Number of Nice Subarrays](https://leetcode.com/problems/count-number-of-nice-subarrays/description/)

## Approach -> 
Same as above

## Code ->
```cpp
class Solution {
public:
    int atmost(vector<int> &nums, int k){
        int l = 0, r = 0, count = 0, numOfSubarr = 0;

        while(r<nums.size()){
            if(nums[r]%2==1) count++;

            while(count>k && l<=r){
                if(nums[l]%2==1) count--;
                l++;
            }
            numOfSubarr += r-l+1;
            r++;
        }
        return numOfSubarr;
    }
    int numberOfSubarrays(vector<int>& nums, int k) {
        return atmost(nums, k) - atmost(nums, k-1);
    }
};
```

# [76. Minimum Window Substring](https://leetcode.com/problems/minimum-window-substring/description/)

## Approach ->
The code implements a sliding window approach to find the minimum window substring of string s that contains all characters from string t. The algorithm initializes two pointers, 'l' and 'r', and a hashmap 'mp' to track the frequency of characters in t. The window expands by moving the right pointer, and when all required characters are found, it shrinks by moving the left pointer. The process continues until the end of the string s. The minimum window size and starting index are updated during the iterations, and the final result is returned as a substring or an empty string if no valid window is found.

[video solution](https://www.youtube.com/watch?v=3Bp3OVD1EGc)

## Code ->
```cpp
class Solution {
public:
    string minWindow(string s, string t) {
        int n = s.size(); 

        // Check if the length of s is less than t, in which case, there is no valid window
        if(n<t.size()) return "";

        // initialize everything. requiredCount = t.size() to keep track of how many elements we require in our window
        int l = 0, r = 0, minWindowSize = INT_MAX, requiredCount = t.size(), start = 0;
        // Create a hashmap 'mp' to track the frequency of characters in t
        unordered_map<char, int> mp;

        // Initialize the hashmap with the frequency of characters in t because we need all characters present in t in our window
        for(int i=0; i<t.size(); i++){
            mp[t[i]]++; 
        }

        // Iterate through the string s
        while(r<n){
            char ch = s[r];

            // when we find an element that is required in our window,
            // i.e. element's frequency is more than 0 in our map,
            // then we can reduce the size of requiredCount
            if(mp[ch]>0) requiredCount--;

            // reduce the frequency of the element from our map
            mp[ch]--;

            // when we have found all the elements that were required in our window, 
            // our main job will be to shrink the window size to make the window minimum possible
            while(requiredCount==0){
                //shrink window

                int curWindowSize = r-l+1;

                // Update minimum window size and starting index if the current window is smaller
                if(curWindowSize<minWindowSize){
                    minWindowSize = curWindowSize;
                    start = l;
                }

                // Move left pointer and update the hashmap
                mp[s[l]]++;
                
                // If we have taken out one element from the window that was required because of moving of l, we increase requiredCount by 1
                if(mp[s[l]]>0) requiredCount++;

                // Move left pointer further
                l++;
            }

            // Move right pointer further
            r++;
        }

        // Return the result substring or an empty string if no valid window is found
        return minWindowSize==INT_MAX ? "" : s.substr(start, minWindowSize);
    }
};
```