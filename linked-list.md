# [876. Middle of the Linked List](https://leetcode.com/problems/middle-of-the-linked-list/description/)
---
## Approaches ->
1. Find the size of the ll and return the size/2 node.
2. Fast-Slow pointer
---
Code ->
```cpp
class Solution {
public:
    ListNode* middleNode(ListNode* head) {
        //ListNode *temp = head;
//         int size=0;
//         while(temp){
//             temp = temp->next;
//             size++;
//         }
        
//         size = size/2 + 1;
//         cout << size;
        
//         temp = head;
//         while(--size)
//             temp = temp->next;
//         return temp;
        
        ListNode *slow=head, *fast=head;
        while(fast && fast->next ){
            slow=slow->next;
            fast=fast->next->next;
        }
        return slow;
    }      
};
```
# [206. Reverse Linked List](https://leetcode.com/problems/reverse-linked-list/description/)
## Iterative Approach Code (TC->O(N) SC->O(1)) ->
```cpp
class Solution {
public:
    ListNode* reverseList(ListNode* head) {
        ListNode* p = NULL, *c = head, *n = NULL;
        while(c){
            n = c->next; // don't forget to assign the value of next here instead at the end to avoid runtime error
            c->next = p;
            p=c;
            c=n;
        }
        return p;
    }
};
```
## [Recursive Approach Code (TC->O(N) SC->O(1)) ->](https://takeuforward.org/data-structure/reverse-a-linked-list/)
```cpp
Node* reverseLinkedList(Node* head) {
    // Base case:
    // If the linked list is empty or has only one node,
    // return the head as it is already reversed.
    if (head == NULL || head->next == NULL) {
        return head;
    }
    
    // Recursive step:
    // Reverse the linked list starting 
    // from the second node (head->next).
    Node* newHead = reverseLinkedList(head->next);
    
    // Save a reference to the node following
    // the current 'head' node.
    Node* front = head->next;
    
    // Make the 'front' node point to the current
    // 'head' node in the reversed order.
    front->next = head;
    
    // Break the link from the current 'head' node
    // to the 'front' node to avoid cycles.
    head->next = NULL;
    
    // Return the 'newHead,' which is the new
    // head of the reversed linked list.
    return newHead;
}
```
# [141. Linked List Cycle](https://leetcode.com/problems/linked-list-cycle/description/)
## Approaches ->
1. Maintain a hashmap. Traverse the ll and find the node in the map. Insert the node in the map.
2. Use two pointers: slow moves one step, fast moves two steps. If a cycle exists, the fast pointer will eventually meet the slow pointer inside the loop. If fast reaches the end (NULL), no cycle exists. This is an efficient O(n) approach with O(1) space.
---
## Code ->
2. 
```cpp
class Solution {
public:
    bool hasCycle(ListNode *head) {
        ListNode *fast = head, *slow = head;
        while(fast && fast->next)
        {
            fast = fast->next->next;
            slow = slow->next;
            if(fast == slow)
                return true;
        }
        return false;
    }
};
```

# [142. Linked List Cycle II](https://leetcode.com/problems/linked-list-cycle-ii/description/)

## Approaches ->
1. We can store nodes in a hash table so that, if a loop exists, the head will encounter the same node again.

2. There are two steps to solve this:

Step 1: Detecting the Loop:
- We use two pointers: slow (tortoise) moves one step at a time, and fast (hare) moves two steps at a time.
- If there's a loop, fast will eventually catch up to slow inside the loop, just like your friend laps you on the track.
- If fast reaches the end (NULL), there's no loop.

Step 2: Finding the Start of the Loop:
- Once fast and slow meet inside the loop, we know there's a cycle.
- Now, we reset fast to the head of the list and keep slow at the meeting point.
- Both pointers now move one step at a time.
- The point where they meet again is the start of the loop.

Why does this work?
- When fast and slow first meet, fast has traveled twice the distance of slow.
- The extra distance fast has traveled is equal to the length of the loop multiplied by some integer (since it's been going around in circles).
- This extra distance also corresponds to the distance from the start of the loop to the meeting point.
- Therefore, the distance from the head to the start of the loop is the same as the distance from the meeting point to the start of the loop (when moving in the same direction).

```cpp
class Solution {
public:
    ListNode *detectCycle(ListNode *head) {
        ListNode* slow = head;
        ListNode* fast = head;

        // Edge case: if the list is empty or has only one node, there can't be a cycle
        if(head == NULL || head->next == NULL) return NULL;

        // Step 1: Detect if there's a cycle using Floyd's Tortoise and Hare algorithm
        while(fast && fast->next) {
            slow = slow->next;       
            fast = fast->next->next; 

            // If they meet, a cycle is detected
            if(fast == slow) break;
        }

        // If fast and slow didn't meet, there's no cycle
        if(fast != slow) return NULL;

        // Step 2: Find the starting node of the cycle
        // Reset fast to the head, keep slow at the meeting point
        fast = head;

        // Move both pointers one step at a time until they meet again
        while(fast != slow) {
            fast = fast->next;
            slow = slow->next;
        }

        // The node where they meet is the start of the cycle
        return fast;
    }
};
```

## Complexity Analysis->
Time Complexity: O(N) 
- The code traverses the entire linked list once, where 'n' is the number of nodes in the list. This traversal has a linear time complexity, O(n).

Space Complexity : O(1) 
- The code uses only a constant amount of additional space, regardless of the linked list's length. This is achieved by using two pointers (slow and fast) to detect the loop without any significant extra memory usage, resulting in constant space complexity, O(1).

# [234. Palindrome Linked List](https://leetcode.com/problems/palindrome-linked-list/description/)

## Approaches ->
1. Use extra space and create an array of ll element and compare in the array.
2. Split the linkedlist into two linked lists from the middle and compare the two linkedlists. (Don't forget to reverse the second linkedlist)

## Code ->
```cpp
class Solution {
public:
    bool isPalindrome(ListNode* head) {
        // If there's only one node, it's a palindrome
        if(head->next == NULL) return true;

        ListNode* slow = head, *fast = head, *temp = head;

        // Move slow by 1 step and fast by 2 steps to reach middle
        while(fast && fast->next){
            slow = slow->next;
            fast = fast->next->next;
        }

        // Reverse the second half of the list starting from 'slow'
        ListNode* prev = NULL, *curr = slow, *nxt = NULL;

        while(curr){
            nxt = curr->next;    
            curr->next = prev;   
            prev = curr;          
            curr = nxt;          
        }

        // Now, 'prev' points to the head of reversed second half
        // Compare first half and reversed second half
        while(prev && temp){
            if(prev->val != temp->val) return false; // Mismatch found
            prev = prev->next;
            temp = temp->next;
        }

        // All nodes matched, it's a palindrome
        return true;
    }
};
```

# [328. Odd Even Linked List](https://leetcode.com/problems/odd-even-linked-list/description/)

## Approaches ->
1. Brute Force: Separate odd and even index nodes into an array using two iterations. Replace the data in the linked list by traversing it again and using the array.
Time complexity: O(n), Space complexity: O(n).
2. Instead of using an external array, rearrange the linked list in-place by changing the links. Traverse the linked list using two pointers, odd and even, starting at the head and head.next. Iterate through the linked list, rearranging the links to connect odd and even nodes alternately. Connect the last odd node to the head of even nodes.
Time complexity: O(n), Space complexity: O(1)

## Code ->
```cpp
class Solution {
public:
    ListNode* oddEvenList(ListNode* head) {
        if(head==NULL || head->next==NULL || head->next->next==NULL) return head;
        ListNode* odd = head, *even = head->next, *toPoint = head->next;

        // connecting all odd nodes and even nodes seperately
        while(odd->next && even->next){
            odd->next = odd->next->next;
            odd = odd->next;
            even->next = even->next->next;
            even = even->next;
        }

        // connecting the end of odd linked list with start of even linked list
        odd->next = toPoint;
        return head;
    }
};
```
# [19. Remove Nth Node From End of List](https://leetcode.com/problems/remove-nth-node-from-end-of-list/description/)
## Approaches -> 
1. We can traverse through the Linked List while maintaining a count of nodes, let’s say in the variable count, and then traversing for the 2nd time for (n – count) nodes to get to the nth node of the list. TC-> O(2n). Keep in mind that there will be 2 different cases for this, first that we might have to delete the first element, in that case simply return head->next. In all other cases, point temp->next to temp->next->next. This question is easy to solve but you will mess the code up if you don't dry run with an example in the whiteboard first. 

2. To delete the Nth node from the end in one pass, we use the two-pointer approach with a dummy node. The dummy node is key—it points to the head and helps handle edge cases cleanly, like when the head itself needs to be deleted (i.e., when n equals the list size).
We initialize fast at the head and slow at the dummy. First, move the fast pointer n steps ahead. Then move both fast and slow one step at a time until fast reaches the end. At this point, slow is just before the node to be deleted (the (n+1)th from the end).
We adjust the links to skip the target node and delete it. This ensures efficient deletion in O(n) time with just one traversal and no need to calculate the list size beforehand. 
To solve this q, first dry run the approach infront of the interviewer then start the implementation otherwise you'd make blunders while coding. 

## Codes ->
1. 
```cpp
class Solution {
public:
    ListNode* removeNthFromEnd(ListNode* head, int n) {

        if(head == NULL) return NULL;

        int size = 0;
        ListNode* temp = head;

        // Step 1: Calculate the total number of nodes
        while(temp){
            size++;
            temp = temp->next;
        }

        // Step 2: Find the position from the start to delete
        size = size - n;
        temp = head;

        // Step 3: If the node to delete is the head
        if(size == 0){
            head = head->next;
            delete temp;
            return head;
        } 

        // Step 4: Traverse to the node just before the one to delete, use pre decrement to do that. 
        while(--size){
            temp = temp->next;
        }

        // Step 5: Remove the target node
        ListNode* toDelete = temp->next;
        temp->next = temp->next->next;
        delete toDelete;

        return head;
    }
};
```

2.
```cpp
class Solution {
public:
    ListNode* removeNthFromEnd(ListNode* head, int n) {
        // Create a dummy node and point its next to head for easier edge case handling
        ListNode* dummy = new ListNode(-1);
        dummy->next = head;

        // Initialize two pointers:  slow at dummy, fast starts at head
        ListNode* slow = dummy;
        ListNode* fast = head;

        // Move fast pointer n steps ahead
        for(int i = 0; i < n; i++) fast = fast->next;

        // Move both pointers until fast reaches the end
        while(fast) {
            fast = fast->next;
            slow = slow->next;
        }

        // If the node to be deleted is the head
        if(slow->next == head) {
            head = head->next;
            delete slow->next;  // Free memory
            return head;
        }

        // Delete the target node
        fast = slow->next;
        slow->next = slow->next->next;
        delete fast;

        return head;
    }
};

```

# [2. Add Two Numbers](https://leetcode.com/problems/add-two-numbers/description/)
## Approach -> 
To solve the problem, simulate the digit-by-digit addition starting from the heads of both linked lists, which represent the least significant digits. Maintain a carry variable to store the overflow from each addition (just like regular addition).
But don’t miss an important edge case — for example:
l1 = [9,9,9,9,9,9,9]
l2 = [9,9,9,9]
Answer: [8,9,9,9,0,0,0,1]

This example shows two critical points:

-> Point 1: Even after one of the lists ends (here, l2 is shorter), you must continue processing the longer list (l1) along with any remaining carry. Notice how even though l2 is done after 4 digits, the carry continues impacting l1's values — making them 0.

-> Point 2: If there's a leftover carry after both lists are fully traversed, don’t forget to add it as a new node at the end of the result (as the final 1 in the example above).

Dry run this example to understand how carry continues to ripple through the remaining digits and affects the final result.

---
## Code ->
```cpp
class Solution {
public:
    ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
        // Initialize variables to keep track of carry, sum, and the value to add to the new node
        int carry = 0, sum = 0, toAdd = 0;

        // Create a dummy node to serve as the head of the result linked list
        ListNode* head = new ListNode(-1);
        
        // Create a pointer to traverse the result linked list
        ListNode* mover = head;

        // Iterate through the input linked lists until both are exhausted
        while (l1 || l2) {
            // Reset sum for each iteration
            sum = 0;

            // Add values from the current nodes of l1 and l2, if available
            if (l1) sum += l1->val;
            if (l2) sum += l2->val;

            // Add the carry from the previous iteration
            sum += carry;

            // Calculate the value to be added to the new node and update the carry
            toAdd = sum % 10;
            carry = sum / 10;

            // Create a new node with the calculated value and append it to the result linked list
            ListNode* temp = new ListNode(toAdd);
            mover->next = temp;
            mover = mover->next;

            // Move to the next nodes in the input linked lists, if available
            if (l1) l1 = l1->next;
            if (l2) l2 = l2->next;
        }

        // If there's a carry remaining after the last iteration, create a new node for it
        if (carry) {
            ListNode* temp = new ListNode(carry);
            mover->next = temp;
            mover = mover->next;
        }

        // Set the next pointer of the last node to NULL to terminate the result linked list
        mover->next = nullptr;

        // Return the actual head of the result linked list (excluding the dummy node)
        return head->next;
    }
};
```

# [237. Delete Node in a Linked List](https://leetcode.com/problems/delete-node-in-a-linked-list/description/)
```cpp
class Solution {
public:
    void deleteNode(ListNode* node) {
        node->val = node->next->val;
        node->next = node->next->next;
    }
};
```

# [160. Intersection of Two Linked Lists](https://leetcode.com/problems/intersection-of-two-linked-lists/description/)

## Approaches ->
1. Brute-Force: Keep any one of the list to check its node present in the other list. Here, we are choosing the second list for this task.
Iterate through the other list. Here, it is the first one. 
Check if the both nodes are the same. If yes, we got our first intersection node.
If not, continue iteration.
If we did not find an intersection node and completed the entire iteration of the second list, then there is no intersection between the provided lists. Hence, return null.
2. Hashing
3. Better Approach: Find the length of both lists.
Find the positive difference between these lengths.
Move the dummy pointer of the larger list by the difference achieved. This makes our search length reduced to a smaller list length.
Move both pointers, each pointing two lists, ahead simultaneously till they collide.
4. Best Approach: Two Pointer Method (Inspired by Length Difference Concept)
This approach is a refined version of the "difference of lengths" idea but implemented more elegantly without explicitly calculating any lengths.
*Intuition:*
The idea is to use two pointers that traverse both linked lists. If the lists intersect, the pointers will eventually meet at the intersection node. If they don’t intersect, both pointers will reach the end (nullptr) at the same time after two passes.
Here’s how:

- Initialize two pointers p1 and p2 at the heads of the two lists.
- Move both forward one step at a time.
- When p1 reaches the end of list A, redirect it to the head of list B.
- When p2 reaches the end of list B, redirect it to the head of list A.
- Eventually, both pointers will either meet at the intersection node or at nullptr.

This works because:
Both pointers will traverse the same number of total nodes: a + b (where a and b are the lengths of the two lists, including the shared part if they intersect).
If there's an intersection, they'll sync up exactly at that node.
If not, they both reach nullptr at the same time.

## Code ->
```cpp
class Solution {
public:
    ListNode *getIntersectionNode(ListNode *headA, ListNode *headB) {
        // Initialize two pointers at the heads of the two lists
        ListNode *p1 = headA;
        ListNode *p2 = headB;

        // Traverse both lists. This is important condition that you traverse till both the p1 and p2 reaches null.
        while(p1 || p2){
            // If p1 reaches end of list A, switch it to head of list B
            if(p1 == NULL) p1 = headB;
            // If p2 reaches end of list B, switch it to head of list A
            if(p2 == NULL) p2 = headA;

            // If the two pointers meet, that's the intersection point, so return the intersecting node
            if(p1 == p2) return p1;

            // Move both pointers forward
            p1 = p1->next;
            p2 = p2->next;
        }

        // If no intersection, both will reach NULL and return NULL
        return p1;
    }
};
```

# [Find pairs with given sum in doubly linked list](https://www.codingninjas.com/studio/problems/find-pairs-with-given-sum-in-doubly-linked-list_1164172?utm_source=striver&utm_medium=website&utm_campaign=a_zcoursetuf&leftPanelTabValue=PROBLEM)

## Approaches ->
1. Brute Force: O(n^2) Time complexity nested loops.
2. Two Pointer: Keep one pointer at start and one at end and move accordingly

## Code ->
```cpp
vector<pair<int, int>> findPairs(Node* head, int k)
{
    // Write your code here.
    Node *st=head;
    Node *en=head;
    while(en->next) en = en->next;
    vector<pair<int, int>> ans;

    while(st && en && st!=en){
        int res = st->data + en->data;
        if(res==k){
            ans.push_back(make_pair(st->data, en->data));
            st = st->next;
            //Check if 'st' and 'en' pointers have converged, if yes, break out of the loop.
            if(st==en) break; // important to check
            en = en->prev;
        }
        else if(res>k){
            en = en->prev;
        }
        else{
            st = st->next;
        }
    }

    return ans;
}
```
# [61. Rotate List](https://leetcode.com/problems/rotate-list/description/)
## Approach ->
- Handle Edge Cases: If the list is empty, has only one node, or k is 0, return the list as-is.
- Count Nodes & Find Tail: Traverse the list to count the number of nodes and identify the last node.
- Calculate Effective Rotation: Since rotating by k (where k >= list length) is equivalent to rotating by k % list length, compute the effective rotation.
- Traverse to the node at the new tail (i.e., at position list length - effective rotation).
- Rotate the List: Set the new head (new tail's next), break the list at the new tail (point it to NULL), and connect the original tail to the original head.


## Code ->
```cpp
class Solution {
public:
    ListNode* rotateRight(ListNode* head, int k) {
        // Edge cases: empty list, single node, or no rotation needed
        if(k == 0 || head == NULL || head->next == NULL) return head;

        // Step 1: Count the number of nodes and find the last node
        int numOfNodes = 0;
        ListNode* temp = head;
        ListNode* lastNode = NULL; 

        while(temp) {
            numOfNodes++;
            lastNode = temp; // Update lastNode 
            temp = temp->next;
        }

        // Step 2: Compute effective rotation (handle cases where k > numOfNodes)
        int effectiveRotation = k % numOfNodes;
        if(effectiveRotation == 0) return head; // No rotation needed

        // Step 3: Find the new tail (node before the new head)
        int stepsToNewTail = numOfNodes - effectiveRotation;
        temp = head;

        // Traverse to the node before the new head
        while(--stepsToNewTail) {
            temp = temp->next;
        }

        // Step 4: Perform rotation
        ListNode* newHead = temp->next; // New head is the next node
        temp->next = NULL; // Break the list at the new tail
        lastNode->next = head; // Connect original tail to original head

        return newHead;
    }
};
```

# [21. Merge Two Sorted Lists](https://leetcode.com/problems/merge-two-sorted-lists/description/)

## Approach ->
This is a fairly simple question but questions like these might seem tough to code if you are out of practice. But there is a very simple way to join the links of the nodes without creating any confusion and that is, you make a dummy node temp and keep pointing it to the node that has the smaller value. Move temp and move list1 and list2 as well. So basically temp is for the conncetion purpose and list1 list2 is for comparision purpose.

## Code ->
```cpp
class Solution {
public:
    ListNode* mergeTwoLists(ListNode* list1, ListNode* list2) {
        // Check if either of the lists is empty.
        if(list1==NULL) return list2;
        if(list2==NULL) return list1;

        // Create a dummy node to simplify handling the merged list.
        ListNode * head = new ListNode(-1);
        ListNode *temp = head;

        // Traverse both lists while they are not empty.
        while(list1 && list2){
            // Compare values of nodes in the two lists.
            if(list1->val < list2->val){
                // Connect the link of smaller value to temp.
                temp->next = list1;
                temp = temp->next;
                list1 = list1->next; 
            }
            else{
                temp->next = list2;
                temp = temp->next;
                list2 = list2->next; 
            }
        }

        // If list1 is not empty, append the remaining nodes.
        while(list1){
            temp->next = list1;
            temp = temp->next;
            list1 = list1->next;
        }
        // If list2 is not empty, append the remaining nodes.
        while(list2){
            temp->next = list2;
            temp = temp->next;
            list2 = list2->next; 
        }
        // Skip the dummy node and return the merged list.
        return head->next;
    }
};
```

# [25. Reverse Nodes in k-Group](https://leetcode.com/problems/reverse-nodes-in-k-group/description/)

## Approach->
This is a hard tagged problem made very simply using recursion..

*Intuition:*
Think of the list as being divided into groups of k nodes.
For each group:
- Reverse the nodes.
- Connect the end of the reversed group to the next group.
But first, we need to make sure there are at least k nodes in the list before reversing. That’s why we calculate the total length at the start.

*Approach:*
1.  Count the length of the list using a simple loop.
2. Use a helper function to:
- Reverse the first k nodes using normal method (iterative)
- Then connect the end of that first k reversed node to the rest of the answer that recursion takes care of.. Just recursively call the function for the next part of the list. You will understand when you see the code.
- In the base case, if fewer than k nodes are left, just return them without reversing.

*Example:*
Input: 1 → 2 → 3 → 4 → 5, k = 2
Reverse 1, 2 → becomes 2 → 1
Reverse 3, 4 → becomes 4 → 3
5 is left alone since it's less than k

Final Output: 2 → 1 → 4 → 3 → 5

## Code ->
```cpp
class Solution {
public:
    // Function to find the total length of the linked list
    int findLength(ListNode* head){
        ListNode* temp = head;
        int l = 0;
        while(temp){
            temp = temp->next;
            l++;
        }
        return l;
    }

    // Helper function to reverse nodes in k-group recursively
    ListNode* helper(ListNode* head, int k, int &size){
        // If list is empty or remaining size is less than k, return head as is
        if(head == NULL) return NULL;
        if(size < k) return head;

        // Pointers for reversing k nodes
        ListNode* prev = NULL;
        ListNode* cur = head;
        ListNode* nxt = NULL;

        int count = 0;

        // Reverse k nodes till cur is not NULL, and up to k nodes
        while(cur && count < k){
            nxt = cur->next;     
            cur->next = prev;    
            prev = cur;          
            cur = nxt;          
            count++;
            size--;              // Decrease remaining size
        }

        // Once the start node was head, now its the end node after reversal. So connect head's next with whatever recursion is giving us
        if(nxt != NULL) head->next = helper(nxt, k, size);

        // prev is the new head of this reversed group, so return new head (prev)
        return prev;
    }

    // Main function
    ListNode* reverseKGroup(ListNode* head, int k) {
        // Find total length of the list and pass it to helper fn. This is to track that if at the end there are less than k nodes left, we do not call recursion for that and simply return the head without reversing. 
        int l = findLength(head);

        return helper(head, k, l);
    }
};
```
## Complexity Analysis->
Time Complexity: O(n)
- We go through each node once for length and once for reversing.

Space Complexity: O(n/k)
- Because of recursion, there can be n/k function calls on the stack in the worst case.

# [ Flatten A Linked List](https://www.geeksforgeeks.org/problems/flattening-a-linked-list/1)

## Approaches ->

1. Brute Force: Make an array, put all the nodes' values in the array, sort the array, make a new Nodes with the sorted values. 
- Time Complexity: O(N*M) + O(N*M log(N*M)) + O(N*M)where N is the length of the linked list along the next pointer and M is the length of the linked list along the child pointer.
- Space Complexity : O(N*M) + O(N*M)

2. Optimal Approach:

Intuition (Simplified Explanation):
Imagine the linked list as a series of vertical towers (sub-linked lists) connected horizontally (via next pointers). Each tower is already sorted from top to bottom (bottom pointer). Our goal is to merge all these towers into one tall, sorted tower.

How?
We use recursion to go deep into the horizontal chain (next pointers) until we reach the last tower. At that point, since there’s nothing left to merge, we return the last tower as is.
Now, as we backtrack step by step:
- We take the current tower (say, Tower A) and merge it with the already flattened result (Tower B, returned from the recursion).
- The merge process combines two sorted vertical towers into one taller sorted tower.
- This merged tower becomes the new "flattened result" for the next backtracking step.

By repeating this merge-at-each-step during backtracking, we eventually combine all towers into a single sorted vertical list.

Key Points:
- Recursion goes deep first (to the end of the horizontal chain).
- Merge happens during backtracking (combining two sorted vertical lists at a time).
- Final result is one fully merged, sorted vertical list.

Note: You sometimes forget that we are dealing with the bottom pointer while coding and keep pointing the dummy node's next to the new nodes, but you have to point it to bottom node and return bottom of dummy node at the end.

This approach works efficiently because we’re merging pre-sorted lists, avoiding the need for extra sorting steps. Read the code to understand this fully-

## Code ->
```cpp
class Solution {
    // Helper function to merge two sorted linked lists (using bottom pointers)
    Node* merge(Node* root, Node* newRoot) {
        // Create a dummy node to simplify the merging process
        Node* dummyNode = new Node(-1);
        Node* res = dummyNode; // 'res' will traverse and build the merged list
    
        // Traverse both lists and merge them in sorted order
        while (root && newRoot) {
            if (root->data < newRoot->data) {
                res->bottom = root;
                res->next = NULL; // Ensure next pointer is not used in the flattened list so all next will point to NULL
                res = root;
                root = root->bottom;
            } else {
                res->bottom = newRoot;
                res->next = NULL; // Ensure next pointer is not used in the flattened list
                res = newRoot;
                newRoot = newRoot->bottom;
            }
        }
    
        // Attach the remaining nodes of either list to our res' bottom
        if (root) res->bottom = root;
        if (newRoot) res->bottom = newRoot;
    
        // Return the merged list (skip the dummy node)
        return dummyNode->bottom;
    }

public:
    // Main function - Function to flatten the linked list
    Node* flatten(Node* root) {
        // Base case: if the list is empty or has only one node, return it as is (because already flat)
        if (root == NULL || root->next == NULL) {
            return root;
        }

        // Recursively flatten the rest of the list (starting from root->next)
        Node* newRoot = flatten(root->next);
        // Merge the current node's vertical-linked list with the flattened list
        root = merge(root, newRoot);
    
        return root;
    }
};
```

# [148. Sort List](https://leetcode.com/problems/sort-list/description/)

## Approach ->
Proper merge sort approach. Read the comments to understand the code, I wrote it beautifully :)

## Code ->
```cpp
class Solution {
public:
    ListNode* merge(ListNode* l1, ListNode* l2){
        // Function to merge two sorted LL

        // Create a temp node that will be your dummy node and a mover that points to temp
        ListNode* temp = new ListNode(-1);
        ListNode* mover = temp;

        // Use l1 and l2 to move forward and compare them and keep adding the lower value nodes to mover's next
        while(l1 && l2){
            if(l1->val < l2->val){
                mover->next = l1;
                l1 = l1->next;
            }
            else{
                mover->next = l2;
                l2 = l2->next;
            }
            // don't forget to move l1, l2 and then mover as well
            mover = mover->next;
        }

        // if either of the remaining l1 or l2 has some nodes left then connect that to mover as well
        if(l1) mover->next = l1;
        if(l2) mover->next = l2;

        // return the sorted array i.e. temp->next
        return temp->next;
    }

    // main function
    ListNode* sortList(ListNode* head) {
        // Let's try solving it like a normal merge sort question

        // Step 0: Base Condition
        if(head==NULL || head->next==NULL) return head;

        // Step 1: Dividing the linked list into two halves.
        ListNode *slow = head, *fast = head, *prev = NULL;
        while(fast && fast->next){
            prev = slow;
            slow = slow->next;
            fast = fast->next->next;
        }

        // Step 2: To properly divide the LL into two seperate LL we have to point the last node of the first LL to null. The second part is already pointing to null.
        prev->next = NULL;

        // Step 3: Sorting the two linked lists by dividing them into further parts
        ListNode* l1 = sortList(head);
        ListNode* l2 = sortList(slow);

        // Step 4: Merge two sorted linked lists
        return merge(l1, l2);
    }

};
```

# [138. Copy List with Random Pointer](https://leetcode.com/problems/copy-list-with-random-pointer/description/)

## Approaches ->
1. Hashmap with extra space. Time Complexity: O(2N), Space Complexity : O(N)+O(N)
2. Optimal: The optimisation will be in removing the extra spaces, i.e, the hashmap used in brute force. We are going to create dummy nodes and place the dummy nodes in between of two actual nodes. Here the dummy node placed after the actual node represents the copy of that actual node. This way, the new nodes carry both the value and connections. This approach can be coded in 3 steps. i. Place dummy nodes between two nodes. ii. Connect the random pointer of the dummy nodes to the previous node's random's next. iii. Separate the real LinkedList and the dummy LinkedList. Dry run after every step and see for corner cases because there's a good chance of going out on a NULL pointer and trying to find its next.



## Codes ->
1. 
```cpp
class Solution {
public:
    Node* copyRandomList(Node* head) {
        // Create a hash map to store the mapping between original nodes and their corresponding new nodes
        unordered_map<Node*, Node*> mp;

        // First iteration: Create new nodes with the same values as the original nodes and store them in the hash map
        Node* temp = head;
        while (temp) {
            // Key: Original node, Value: New node with the same value
            mp[temp] = new Node(temp->val);
            temp = temp->next;
        }

        // Reset temp to the head of the original list for the second iteration
        temp = head;

        // Second iteration: Link next and random pointers of new nodes based on the mapping in the hash map
        while (temp) {
            // Link the next and random pointers of the new nodes using the hash map
            mp[temp]->next = mp[temp->next];
            mp[temp]->random = mp[temp->random];
            temp = temp->next;
        }

        // Return the head of the new linked list (value node of the original head node)
        return mp[head];
    }
};
```

2. 
```cpp
class Solution {
public:
    Node* copyRandomList(Node* head) {
        if (!head) return NULL;  // Edge case: empty list

        // Step 1: Insert dummy (copied) nodes after each original node
        Node* temp = head;
        while (temp) {
            Node* dummyNode = new Node(temp->val);  // Create copy
            dummyNode->next = temp->next;          // Insert copy after original
            temp->next = dummyNode;                 
            temp = temp->next->next;                
        }

        // Step 2: Set random pointers for copied nodes
        Node* prev = head;          // Track original nodes
        temp = head->next;          // Track copied nodes

        while (temp) {
            // If original's random exists, point copy's random to its copy. This conditional check if very important because an original node's random might point to NULL, and you cannot check for a NULL's next.
            temp->random = (prev->random) ? prev->random->next : NULL;
            
            if (temp->next==NULL) break;  // End of list
            temp = temp->next->next; 
            prev = prev->next->next; 
        }

        // Step 3: Separate original and copied lists
        prev = head;
        temp = head->next;
        Node* ans = temp;  // Head of the copied list
        while (temp) {
            prev->next = prev->next->next;  /
            temp->next = (temp->next) ? temp->next->next : NULL;  // Link copied list
            prev = prev->next;              
            temp = temp->next;              
        }
        
        return ans;  // Return head of the copied list
    }
};
```

# [1472. Design Browser History](https://leetcode.com/problems/design-browser-history/)

## Code ->
```cpp
// Define a structure for the linkedlist to represent pages in the browser history. You have to do this yourself in the question, there is no template already made for it. 
struct Node{
    string val;
    Node *prev;
    Node *next;

    // Create the constructor for ease of use
    Node(string value=""){
        val = value;
        prev = NULL;
        next = NULL;
    }
};

class BrowserHistory {
public:

    Node *currentPage;

    // Create a new node for the string homepage and set it as the current page
    BrowserHistory(string homepage) {
        currentPage = new Node(homepage);
    }
    
    // Function to visit a new URL, creating a new page and updating the history
    void visit(string url) {
        // Create a new node for the visited URL
        Node *newPage = new Node(url);
        // Update pointers to link the new page to the current page
        currentPage->next = newPage;
        newPage->prev = currentPage;
        // Update the current page to the newly visited page
        currentPage = newPage;
    }
    
    // Function to move back in history by a given number of steps
    string back(int steps) {
        while(steps--){
            // break if we already reached the first page
            if(currentPage->prev==NULL) break;
            currentPage = currentPage->prev;
        }
        return currentPage->val;
    }
    
    // Function to move forward in history by a given number of steps
    string forward(int steps) {
        while(steps--){
            // break if we already reached the last page
            if(currentPage->next==NULL) break;
            currentPage = currentPage->next;
        }
        return currentPage->val;
    }
};
```


# [23. Merge k Sorted Lists](https://leetcode.com/problems/merge-k-sorted-lists/description/)

## Code ->
```cpp

class Solution {
public:
    ListNode *mergeKLists(vector<ListNode *> &lists) {
        if(lists.empty()){
            return nullptr;
        }
        while(lists.size() > 1){
            lists.push_back(mergeTwoLists(lists[0], lists[1]));
            lists.erase(lists.begin());
            lists.erase(lists.begin());
        }
        return lists.front();
    }
    ListNode *mergeTwoLists(ListNode *l1, ListNode *l2) {
        if(l1 == nullptr){
            return l2;
        }
        if(l2 == nullptr){
            return l1;
        }
        if(l1->val <= l2->val){
            l1->next = mergeTwoLists(l1->next, l2);
            return l1;
        }
        else{
            l2->next = mergeTwoLists(l1, l2->next);
            return l2;
        }
    }
};
```

# [LRU Cache](https://leetcode.com/problems/lru-cache/)

## Problem Explanation
Before reading this, see the example and understand the question better. Then try on your own because this is one of those questions which need practice.
The Least Recently Used (LRU) cache is a data structure that maintains items in order of their usage. When the cache reaches its capacity, it evicts the least recently used item before adding a new one. We need to implement this with O(1) average time complexity for both `get` and `put` operations.

## Intuition
To achieve O(1) operations:
- We need fast lookups (hash map)
- We need to maintain order of usage (doubly linked list)
  
The hash map gives us O(1) access to any node, while the doubly linked list allows us to move nodes to the front (most recently used) or remove from the back (least recently used) in O(1) time.

## Approach
1. **Data Structures**:
   - `unordered_map<int, Node*>` for O(1) lookups by key
   - Doubly linked list to maintain usage order with:
     - `frontNode` (dummy head) pointing to least recently used
     - `backNode` (dummy tail) pointing to most recently used

2. **Operations**:
   - `get(key)`: 
     - If key exists, move node to back (most recently used) and return value
     - Else return -1
   - `put(key, value)`:
     - If key exists, update value and move to back
     - If at capacity, remove node at front (LRU) before adding new node
     - Add new node at back

## Solution Code

```cpp
// Node structure for doubly linked list
struct Node {
    int key;
    int value;
    Node* front; // pointer to next node
    Node* back;  // pointer to previous node

    Node(int key, int value) {
        this->key = key;
        this->value = value;
        front = nullptr;
        back = nullptr;
    }
};

class LRUCache {
private:
    int cap;                      // Maximum capacity of cache
    unordered_map<int, Node*> mp; // Hash map for O(1) access
    Node* frontNode;              // Dummy head (points to LRU)
    Node* backNode;               // Dummy tail (points to MRU)

public:
    // Initialize with given capacity
    LRUCache(int capacity) {
        cap = capacity;
        // Create dummy nodes to simplify edge cases
        frontNode = new Node(-1, -1);
        backNode = new Node(-1, -1);
        // Connect them initially
        frontNode->front = backNode;
        backNode->back = frontNode;
    }

    // Remove a node from its current position in the list
    void deleteNode(Node* node) {
        // Connect previous and next nodes
        node->back->front = node->front;
        node->front->back = node->back;
        // Free memory (commented out to avoid double free in attachAtEnd)
        // delete node;
    }

    // Move node to back (most recently used position)
    void attachAtEnd(Node* node) {
        // First remove from current position if needed
        if (node->back) node->back->front = node->front;
        if (node->front) node->front->back = node->back;
        
        // Insert just before backNode
        Node* prev = backNode->back;
        prev->front = node;
        node->back = prev;
        node->front = backNode;
        backNode->back = node;
    }

    // Get value for key if exists, else return -1
    int get(int key) {
        if (!mp.count(key)) {
            return -1; // Key not found
        }
        Node* node = mp[key];
        // Move accessed node to most recently used position
        deleteNode(node);
        attachAtEnd(node);
        return node->value;
    }

    // Insert or update key-value pair
    void put(int key, int value) {
        if (mp.count(key)) {
            // Key exists - update value and move to MRU
            Node* node = mp[key];
            node->value = value;
            deleteNode(node);
            attachAtEnd(node);
        } else {
            // Check if we need to evict LRU item
            if (mp.size() >= cap) {
                // Remove from map and delete LRU node
                Node* lru = frontNode->front;
                mp.erase(lru->key);
                deleteNode(lru);
                delete lru; // Free memory
            }
            // Create new node and add to MRU position
            Node* newNode = new Node(key, value);
            attachAtEnd(newNode);
            mp[key] = newNode; // Add to map
        }
    }
};
```

## Complexity Analysis

**Time Complexity**:
- `get(key)`: O(1) average (hash map lookup + constant time linked list operations)
- `put(key, value)`: O(1) average (same as get plus constant time insertions/deletions)

**Space Complexity**: O(capacity) - We store at most 'capacity' nodes in the map and linked list.

The solution efficiently combines a hash map for fast lookups with a doubly linked list for maintaining usage order, meeting all requirements for an LRU cache implementation.

# [LFU Cache](https://leetcode.com/problems/lfu-cache/)
To solve...