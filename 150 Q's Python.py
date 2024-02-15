from cmath import inf
import collections
import heapq
from itertools import chain
import math
import random
import string
import sys
from typing import Generator, List, Optional
from collections import Counter, defaultdict, deque

def merge(nums1: List[int], m: int, nums2: List[int], n: int) -> None:
    for j in range(n):
        nums1[m+j] = nums2[j]
    nums1.sort()

def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
    a, b, write_index = m-1, n-1, m + n - 1

    while b >= 0:
        if a >= 0 and nums1[a] > nums2[b]:
            nums1[write_index] = nums1[a]
            a -= 1
        else:
            nums1[write_index] = nums2[b]
            b -= 1

        write_index -= 1


def removeElement(nums: List[int], val: int) -> int:
    i = 0
    j = 0
    while i <= len(nums)-1:
        if nums[i] != val:
            nums[j] = nums[i]
            j += 1
            i += 1
        else:
            i += 1
    return j

def removeElement(self, nums: List[int], val: int) -> int:
    index = 0
    for i in range(len(nums)):
        if nums[i] != val:
            nums[index] = nums[i]
            index += 1
    return index

def removeDuplicates(nums: List[int]) -> int:
    j = 1
    for i in range(1, len(nums)):
        if nums[i] != nums[i - 1]:
            nums[j] = nums[i]
            j += 1
    return j

def removeDuplicates(nums: List[int]) -> int:
    j = 0
    i = 1
    count = 1
    while i < len(nums):
        if nums[i] == nums[j] and count<2:
            count += 1
            j +=1
            nums[j] = nums[i]
        elif nums[i]==nums[j] and count>=2:
            count+=1
        else:
            count=1
            j+=1
            nums[j]=nums[i]
        i+=1

    return j+1

class Solution(object):
    def removeDuplicates(self, nums):
        
        
        k = 0
        
        for i in nums:
            
            if k < 2 or i != nums[k - 2]:
                nums[k] = i
                k += 1
        return k       

def majorityElement(nums: List[int]) -> int:    
    count = 0
    candidate = 0
    
    for num in nums:
        if count == 0:
            candidate = num
        
        if num == candidate:
            count += 1
        else:
            count -= 1
    
    return candidate

class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        nums.sort()
        n = len(nums)
        return nums[n//2]

def majorityElement(nums: List[int]) -> int:    
    n = len(nums)
    m = defaultdict(int)

    for num in nums:
        m[num] += 1

    n = n // 2
    for key, value in m.items():
        if value > n:
            return key

    return 0

def rotate(nums: List[int], k: int) -> None:
    n=len(nums)
    k = k%n
    nums[:n-k]=nums[:n-k][::-1]
    nums[n-k:]=nums[n-k:][::-1]
    nums[:]=nums[::-1]

def rotate(nums: List[int], k: int) -> None:
    for i in range(k%len(nums)):
        a=nums.pop()
        nums.insert(0,a)

def maxProfit(prices: List[int]) -> int:
    best = 0
    profit = []
    j = 0
    for i in range(0,len(prices)):
        for j in range (i+1,len(prices)):
            if prices[j]-prices[i] > best:
                best = prices[j]-prices[i]
        profit.insert(0,best)
        best = 0
    return(max(profit))

def maxProfit(prices: List[int]) -> int:
    if len(prices) == 0: return 0
    else:
        profit = 0
        minBuy = prices[0]
        for i in range(len(prices)):
            profit = max(prices[i] - minBuy, profit)
            minBuy = min(minBuy, prices[i])
        return profit

def maxProfit(self, p):
    res, min_so_far = 0, math.inf
    
    for p1 in p:
            res, min_so_far = max(res, p1 - min_so_far), min(min_so_far, p1)
    
    return res


def maxProfit(prices: List[int]) -> int:
    if len(prices) == 0: return 0
    else:
        Total = 0
        profit = 0
        minBuy = prices[0]
        for i in range(len(prices)):
            profit = max(prices[i] - minBuy, profit)
            minBuy = min(minBuy, prices[i])
            if prices[min(i+1,len(prices)-1)] < prices[i]:
                Total += profit
                minBuy = prices[i+1]
                profit = 0
        Total += profit
        return Total
    
def maxProfit(self, prices: List[int]) -> int:
    
    profit_from_price_gain = 0
    for idx in range( len(prices)-1 ):
        
        if prices[idx] < prices[idx+1]:
            profit_from_price_gain += ( prices[idx+1] - prices[idx])
            
    return profit_from_price_gain
    
def canJump(nums: List[int]) -> bool:
    PossibleJumps = nums[0]
    for i in range(1,len(nums)):
        if PossibleJumps == 0:
            return False
        PossibleJumps -= 1
        PossibleJumps = max(PossibleJumps, nums[i])
    return True


def canJump(self, nums: List[int]) -> bool:
    reachableIndex = 0
    for curr in range(len(nums)):
        if curr + nums[curr] >= reachableIndex:
            reachableIndex = curr + nums[curr]
        if curr == reachableIndex:
            break
            
    return reachableIndex >= len(nums) - 1

def jump(nums: List[int]) -> int:
    ans = 0
    end = 0
    farthest = 0
    for i in range(len(nums) - 1):
        farthest = max(farthest, i + nums[i])
        if farthest >= len(nums) - 1:
            ans += 1
            break
        if i == end:     
            ans += 1       
            end = farthest  

    return ans

def hIndex(citations: List[int]) -> int:
    citations.sort()
    current = 0
    for i in range (len(citations)):
        if citations[i] > len(citations) - i:
            return max(current, (len(citations) - i))
        current = citations[i]
    return current

class RandomizedSet:

    def __init__(self):
        self.list = []

    def insert(self, val: int) -> bool:
        for i in range(len(self.list)):
            if self.list[i] == val:
                return False
        self.list.insert(0,val)
        return True
        

    def remove(self, val: int) -> bool:
        for i in range(len(self.list)):
            if self.list[i] == val:
                self.list.pop(i)
                return True
        return False

    def getRandom(self) -> int:
        return random.choice(self.list)
    
class RandomizedSet:

    def __init__(self):
        self.data_map = {} 
        self.data = [] 

    def insert(self, val: int) -> bool:
        
        
        
        
        if val in self.data_map:
            return False

        
        
        
        self.data_map[val] = len(self.data)

        
        self.data.append(val)
        
        return True

    def remove(self, val: int) -> bool:
        
        
        if not val in self.data_map:
            return False

        
        
        
        
        
        last_elem_in_list = self.data[-1]
        index_of_elem_to_remove = self.data_map[val]

        self.data_map[last_elem_in_list] = index_of_elem_to_remove
        self.data[index_of_elem_to_remove] = last_elem_in_list

        
        
        self.data[-1] = val

        
        self.data.pop()

        
        self.data_map.pop(val)
        return True

    def getRandom(self) -> int:
        return random.choice(self.data)

def productExceptSelf(nums: List[int]) -> List[int]:
    length=len(nums)
    sol=[1]*length
    pre = 1
    post = 1
    for i in range(length):
        sol[i] *= pre
        pre = pre*nums[i]
        sol[length-i-1] *= post
        post = post*nums[length-i-1]
    return(sol)

def canCompleteCircuit(gas: List[int], cost: List[int]) -> int:
    Tank = 0
    j = 0
    n = len(gas)
    for i in range (n):
        while j < n and Tank >= 0:
            Tank = Tank + gas[(i + j)%n] - cost[(i + j)%n]
            j += 1
        if Tank >= 0:
            return i
        else:
            Tank = 0
            j = 0
    return -1

def canCompleteCircuit(gas: List[int], cost: List[int]) -> int:
    if (sum(gas) - sum(cost) < 0):
        return -1
    Tank = 0
    j = 0
    n = len(gas)
    for i in range (n):
        Tank += gas[i] - cost [i]
        if Tank < 0:
            j = i+1
            Tank = 0
    return j

def canCompleteCircuit(self, gas: List[int], cost: List[int]) -> int:
    if sum(gas) < sum(cost): return -1
    tank = idx = 0
    for i in range(len(gas)):
        tank+= gas[i]-cost[i] 
        if tank < 0: tank, idx = 0, i+1
    return idx 

    return -1

def candy(ratings: List[int]) -> int:
    length=len(ratings)
    minimum = [1]*length
    for i in range (1,length):
        if ratings[i] > ratings[i-1]:
            minimum[i] += minimum[i-1] 
    for j in range (1,length):
        if ratings[length-1-j] > ratings[length-j] and minimum[length-1-j] <= minimum[length-j]:
            minimum[length-1-j] = minimum[length-j] + 1
    return sum(minimum)

def romanToInt(s: str) -> int:
    numerals = {'I': 1, 'V': 5, 'X': 10,'L': 50, 'C': 100, 'D': 500,'M':1000}
    length = len(s)
    Total = numerals[s[length-1]]
    for i in range(length-2,-1,-1):
        if numerals[s[i+1]] > numerals[s[i]]:
            Total -= numerals[s[i]]
        else:
            Total += numerals[s[i]]
    return Total

def romanToInt(s):
    translations = {
        "I": 1,
        "V": 5,
        "X": 10,
        "L": 50,
        "C": 100,
        "D": 500,
        "M": 1000
    }
    number = 0
    s = s.replace("IV", "IIII").replace("IX", "VIIII")
    s = s.replace("XL", "XXXX").replace("XC", "LXXXX")
    s = s.replace("CD", "CCCC").replace("CM", "DCCCC")
    for char in s:
        number += translations[char]
    return number, s
romanToInt('IVV')

def trap(height: List[int]) -> int:
    
    if len(height)<= 2:
        return 0
    
    ans = 0
    
    
    i = 1
    j = len(height) - 1
    
    
    lmax = height[0]
    rmax = height[-1]
    
    while i <=j:
        
        if height[i] > lmax:
            lmax = height[i]
        if height[j] > rmax:
            rmax = height[j]
        
        
        if lmax <= rmax:
            ans += lmax - height[i]
            i += 1
            
        
        else:
            ans += rmax - height[j]
            j -= 1
            
    return ans

def intToRoman(num: int) -> str:
    numerals = {'M':1000,'CM': 900, 'D': 500,'CD': 400, 'C': 100, 'XC':90 ,'L': 50, 'XL':40, 'X': 10, 'IX': 9, 'V': 5, 'IV':4,'I': 1}
    String = ''
    for key, value in numerals.items():
        if num - value >= 0:
            n = math.floor(num/value)
            num -= value*n
            String += key*n
    return String

def lengthOfLastWord(s: str) -> int:
    j = 0
    start = 0
    for i in s[::-1]:
        if i != ' ':
            j +=1
        if i != ' ' and start == 0:
            start = 1 
            j = 1
        if i == ' ' and start ==1 :
            return j 
    return j

def lengthOfLastWord(s: str) -> int:
    wordlist = s.split()
    if wordlist:
        return len(wordlist[-1])
    return 0

def longestCommonPrefix(strs: List[str]) -> str:
    Prefix = ''
    k = 0
    while k < len(strs[0]):
        for i in range(1,len(strs)):
            if k == len(strs[i]) or strs[i][k] != strs[0][k]:
                return Prefix
        Prefix += strs[0][k]
        k +=1
            
    return Prefix


def longestCommonPrefix(v: List[str]) -> str:
    ans=""
    v=sorted(v)
    first=v[0]
    last=v[-1]
    for i in range(min(len(first),len(last))):
        if(first[i]!=last[i]):
            return ans
        ans+=first[i]
    return ans

def reverseWords(s: str) -> str:
    return(" ".join(s.split()[::-1]))

def convert(s: str, numRows: int) -> str:
    if numRows == 1:
        return s
    Row = 0
    Concat = ['']*numRows
    increase = 1
    for i in s:
        Concat[Row] += i
        Row += increase
        if Row == numRows -1 :
            increase = -1
        if Row == 0:
            increase = 1
    return(''.join(Concat))

def strStr(haystack: str, needle: str) -> int:
    index = 0
    j = 0
    for i in haystack:
        if i == needle[j]:
            while j < len(needle):
                if j == len(needle) - 1:
                    return index - j
                j +=1
                index += 1
                if index >= len(haystack) or haystack[index] != needle[j]:
                    index = index - j
                    j = 0
                    break
                
        index +=1
    return -1

def strStr(haystack: str, needle: str) -> int:
    return haystack.find(needle)

def strStr(haystack: str, needle: str) -> int:
    n, h = len(needle), len(haystack)
    hash_n = hash(needle)
    for i in range(h-n+1):
        if hash(haystack[i:i+n]) == hash_n:
            return i
    return -1

def strStr(haystack, needle):
    def f(c):
        return ord(c)-ord('A')

    n, h, d, m = len(needle), len(haystack), ord('z')-ord('A')+1, sys.maxint 
    if n > h: return -1
    nd, hash_n, hash_h = d**(n-1), 0, 0   
    for i in range(n):
        hash_n = (d*hash_n+f(needle[i]))%m
        hash_h = (d*hash_h+f(haystack[i]))%m            
    if hash_n == hash_h: return 0        
    for i in range(1, h-n+1):
        hash_h = (d*(hash_h-f(haystack[i-1])*nd)+f(haystack[i+n-1]))%m    
        if hash_n == hash_h: return i
    return -1

def strStr(haystack, needle):
	n, h = len(needle), len(haystack)
	i, j, nxt = 1, 0, [-1]+[0]*n
	while i < n:                                
		if j == -1 or needle[i] == needle[j]:   
			i += 1
			j += 1
			nxt[i] = j
		else:
			j = nxt[j]
	i = j = 0
	while i < h and j < n:
		if j == -1 or haystack[i] == needle[j]:
			i += 1
			j += 1
		else:
			j = nxt[j]
	return i-j if j == n else -1

def fullJustify(words: List[str], maxWidth: int) -> List[str]:
    output = []
    string = ''
    j=0
    for i in range(len(words)):
        if string == '':
            string += words[i]
        else:
            if maxWidth - len(string) - len(words[i]) - 1 >=0:
                string += ' ' + words[i]
            else:
                line = string.split()
                if len(string) <maxWidth and len(line) >1:
                    spaces = ' '*(((maxWidth - len(string))//(max(len(line)-1,1)))+1)
                    string = spaces.join(line)
                if len(line) == 1:
                    string += ' '*(maxWidth-len(string))  
                while len(string) < maxWidth:
                    if string[j] == ' ':
                        string = string[:j] + " " + string[j:]
                        j += 1
                    j += 1
                j = 0
                output.append(string)
                string = words[i]
    string += ' '*(maxWidth-len(string))
    output.append(string)
    return output
        
def fullJustify(words: List[str], maxWidth: int) -> List[str]:
    output, line, letters = [],[],0
    for w in words:
        if letters + len(w) + len(line) > maxWidth:
            for i in range(maxWidth - letters):
                line[i%(len(line)-1 or 1)] += " "
            output.append(''.join(line))
            line, letters = [],0
        line += [w]
        letters += len(w)
    return output + [' '.join(line).ljust(maxWidth)]
        

                
def fullJustify(words: List[str], maxWidth: int) -> List[str]:
    output, line, letters = [],[],0
    for w in words:
        if letters + len(w) + len(line) > maxWidth:
            for i in range(maxWidth - letters):
                line[i%(len(line)-1 or 1)] += " "
            output.append(''.join(line))
            line, letters = [],0
        line += [w]
        letters += len(w)
    string = ' '.join(line)
    string +=' '*(maxWidth - len(string))
    return output + [string]

def isPalindrome(s: str) -> bool:
    return True if ''.join(char for string in s.split() for char in string if char.isalnum()).lower()[::-1] == ''.join(char for string in s.split() for char in string if char.isalnum()).lower() else False

def isPalindrome(s: str) -> bool:
    s = ''.join(char for char in s if char.isalnum()).lower()
    return True if s == s[::-1] else False

def isPalindrome(s: str) -> bool:
    s = ''.join(char for char in s.lower() if char.isalnum())
    return s == s[::-1]

def isPalindrome(s: str) -> bool:
    i, j = 0, len(s) - 1
    while i < j:
        a, b = s[i].lower(), s[j].lower()
        if a.isalnum() and b.isalnum():
            if a != b: return False
            else:
                i, j = i + 1, j - 1
                continue
        i, j = i + (not a.isalnum()), j - (not b.isalnum())
    return True

def isSubsequence(s: str, t: str) -> bool:
    if len(s) == 0:
        return True
    i = 0
    for l in t:
        if l == s[i]:
            i += 1
            if i == len(s):
                return True
    return False

def isSubsequence(s: str, t: str) -> bool:
    for c in s:
        i = t.find(c)
        if i == -1:    return False
        else:   t = t[i+1:]
    return True

def twoSum(numbers: List[int], target: int) -> List[int]:
    i = 0
    j = len(numbers) - 1
    while i < j:
        if numbers[j] + numbers[i] == target:
            return [i+1,j+1]
        elif numbers[j] + numbers[i] > target:
            j -= 1
        else:
            i += 1


def twoSum(numbers: List[int], target: int) -> List[int]:
    dic={}
    for key,val in enumerate(numbers):
        if val in dic:
            return dic[val],key
        else:
            dic[target-val]=key


def maxArea(height: List[int]) -> int:
    i = 0
    distance = len(height) -1
    j = distance
    lheight = height[i]
    rheight = height[distance]
    maxwater = (distance)*min(rheight,lheight)
    while distance >0:
        maxwater = max((distance)*min(rheight,lheight),maxwater)
        if lheight > rheight:
            j -= 1
            rheight = height[j]
        else:
            i +=1
            lheight = height[i]
        distance -= 1
    return maxwater

def threeSum(self, nums: List[int]) -> List[List[int]]:
    nums.sort()
    output = []
    for i in range(len(nums)-2):
        target = -nums[i]
        numbers = nums[i+1:]
        j = 0
        k = len(numbers) - 1
        while j < k:
            if numbers[j] + numbers[k] == target:
                if [nums[i], numbers[j], numbers[k]] not in output:
                    output.append([nums[i], numbers[j], numbers[k]])
                k -=1
                j +=1
            elif numbers[j] + numbers[k] > target:
                k -= 1
            else:
                j += 1
    return output

def threeSum(nums: List[int]) -> List[List[int]]:

	res = set()

	
	n, p, z = [], [], []
	for num in nums:
		if num > 0:
			p.append(num)
		elif num < 0: 
			n.append(num)
		else:
			z.append(num)

	
	N, P = set(n), set(p)

	
	
	if z:
		for num in P:
			if -1*num in N:
				res.add((-1*num, 0, num))

	
	if len(z) >= 3:
		res.add((0,0,0))

	
	
	for i in range(len(n)):
		for j in range(i+1,len(n)):
			target = -1*(n[i]+n[j])
			if target in P:
				res.add(tuple(sorted([n[i],n[j],target])))

	
	
	for i in range(len(p)):
		for j in range(i+1,len(p)):
			target = -1*(p[i]+p[j])
			if target in N:
				res.add(tuple(sorted([p[i],p[j],target])))

	return res

def threeSum(nums: List[int]) -> List[List[int]]:
    nums.sort()
    output = []
    for i in range(len(nums)-2):
        if i > 0 and nums[i] == nums[i-1]:
            continue
        target = -nums[i]
        j = i+1
        k = len(nums) - 1
        while j < k:
            if nums[j] + nums[k] == target:
                output.append([nums[i], nums[j], nums[k]])
                while j + 1 < len(nums) and nums[j+1] == nums[j]:
                    j += 1
                while k - 1 >= 0 and nums[k-1] == nums[k]:
                    k-=1
                j +=1
                k -=1
            elif nums[j] + nums[k] > target:
                k -= 1
            else:
                j += 1
    return output

def minSubArrayLen(target: int, nums: List[int]) -> int:
    if len(nums) == 0: return 0
    i, j = 0, 0
    c, t = float("inf"), nums[0]
    while j <= len(nums)-1:
        if t < target:
            j += 1
            if j <= len(nums)-1:
                t += nums[j]
        elif t >= target:
            c = min(j - i + 1, c)
            t -= nums[i]
            i += 1
    return c if c != float("inf") else 0

def lengthOfLongestSubstring(s: str) -> int:
    cur = ''
    output = 0
    for i in range(len(s)):
        if s[i] in cur:
            if len(cur) > output:
                output = len(cur)
            cur = cur[cur.find(s[i])+1:]
        cur += s[i]
    return max(len(cur),output)
        

def lengthOfLongestSubstring(s: str) -> int:
    seen = {}
    l = 0
    output = 0
    for r in range(len(s)):
        if s[r] not in seen:
            output = max(output,r-l+1)
        else:
            if seen[s[r]] < l:
                output = max(output,r-l+1)
            else:
                l = seen[s[r]] + 1
        seen[s[r]] = r
    return output

def findSubstring(self, s: str, words: List[str]) -> List[int]:
    if not words: return []
    LS, M, N, C = len(s), len(words), len(words[0]), collections.Counter(words)
    
    ans = []

    for s_idx in range(LS-M*N+1):
        temp = []
        for subarray_idx in range(s_idx, s_idx+M*N, N):
            temp.append(s[subarray_idx:subarray_idx+N])
            
            
        count_subarray = Counter(temp)
        
        if count_subarray == C:
            ans.append(s_idx)
        

    return ans

def findSubstring(s: str, words: List[str]) -> List[int]:
    length = len(words[0])
    word_count = Counter(words)
    indexes = []

    for i in range(length):
        start = i
        window = defaultdict(int)
        words_used = 0

        for j in range(i, len(s) - length + 1, length):
            word = s[j:j + length]

            if word not in word_count:
                start = j + length
                window = defaultdict(int)
                words_used = 0
                continue

            words_used += 1
            window[word] += 1

            while window[word] > word_count[word]:
                window[s[start:start + length]] -= 1
                start += length
                words_used -= 1

            if words_used == len(words):
                indexes.append(start)

    return indexes

def minWindow(s: str, t: str) -> str:
    window = defaultdict(int)
    letters = Counter(t)
    left = 0
    output = ''
    equal = 0
    for i in range(len(s)):
        if s[i] in letters:
            window[s[i]] +=1
            if window[s[i]] == letters[s[i]]:
                equal += 1
        if equal == len(letters):
            for j in range(left, i+1):
                if s[j] in letters:
                    window[s[j]] -=1
                    if window[s[j]] < letters[s[j]]:
                        left = j
                        equal -=1
                        break
            if len(output) == 0:
                output = s[left:i+1]
            elif len(output) > (i + 1 - left):
                output =  s[left:i+1]
            left +=1
    return output
            
from collections import Counter

class Solution:
    def minWindow(s: str, t: str) -> str:
        
        if not s or not t or len(s) < len(t):
            return ''
        
        t_counter = Counter(t)
        chars = len(t_counter.keys())
        
        s_counter = Counter()
        matches = 0
        
        answer = ''
        
        i = 0
        j = -1 
        
        while i < len(s):
            
            
            if matches < chars:
                
                
                if j == len(s) - 1:
                    return answer
                
                j += 1
                s_counter[s[j]] += 1
                if t_counter[s[j]] > 0 and s_counter[s[j]] == t_counter[s[j]]:
                    matches += 1

            
            else:
                s_counter[s[i]] -= 1
                if t_counter[s[i]] > 0 and s_counter[s[i]] == t_counter[s[i]] - 1:
                    matches -= 1
                i += 1
                
            
            if matches == chars:
                if not answer:
                    answer = s[i:j+1]
                elif (j - i + 1) < len(answer):
                    answer = s[i:j+1]
        
        return answer

def isValidSudoku(board: List[List[str]]) -> bool:
    for i in range (9):
        check = Counter(board[i])
        for j in range (1,10):
            if check[str(j)] > 1:
                return False
    for i in range(9):
        current = [board[j][i] for j in range(9)]
        check = Counter(current)
        for j in range (1,10):
            if check[str(j)] > 1:
                return False
    for i in range(3):
        for j in range(3):
            current = [board[x][y] for x in range(j*3,j*3 + 3)for y in range(3*i, i*3 + 3)]
            check = Counter(current)
            for k in range (1,10):
                if check[str(k)] > 1:
                    return False

    return True


def isValidSudoku(board):
    res = []
    for i in range(9):
        for j in range(9):
            element = board[i][j]
            if element != '.':
                res += [(i, element), (element, j), (i // 3, j // 3, element)]
    return len(res) == len(set(res))

def spiralOrder(matrix: List[List[int]]) -> List[int]:
    output = []
    BB = len(matrix)
    RB = len(matrix[0])
    UB, LB = 0, 0
    i,j = 0,0
    while len(output) < len(matrix[0])*len(matrix):
        while j < RB:
            output += [matrix[i][j]]
            j += 1
        j -=1
        i += 1
        UB +=1
        if len(output) == len(matrix[0])*len(matrix):
            return output
        while i < BB:
            output += [matrix[i][j]]
            i += 1
        i -=1
        j -=1
        RB -=1
        if len(output) == len(matrix[0])*len(matrix):
            return output
        while j >= LB:
            output += [matrix[i][j]]
            j -= 1
        j +=1
        i -= 1
        BB -=1
        if len(output) == len(matrix[0])*len(matrix):
            return output
        while i >= UB:
            output += [matrix[i][j]]
            i -= 1
        i +=1
        j +=1
        LB += 1
        print(i,j,RB)
    return output

def spiralOrder(matrix: List[List[int]]) -> List[int]:
    res = []
    row_begin = 0
    col_begin = 0
    row_end = len(matrix)-1 
    col_end = len(matrix[0])-1
    while len(res) < len(matrix[0])*len(matrix):
        for i in range(col_begin,col_end+1):
            res.append(matrix[row_begin][i])
        row_begin += 1
        for i in range(row_begin,row_end+1):
            res.append(matrix[i][col_end])
        col_end -= 1
        if (row_begin <= row_end):
            for i in range(col_end,col_begin-1,-1):
                res.append(matrix[row_end][i])
            row_end -= 1
        if (col_begin <= col_end):
            for i in range(row_end,row_begin-1,-1):
                res.append(matrix[i][col_begin])
            col_begin += 1
    return res

def rotate(matrix: List[List[int]]) -> None:
    length = len(matrix)
    for i in range(length):
        matrix.append([])
        for j in range(length):
            matrix[length+i].insert(0,matrix[j][i])
    del matrix[:length]


def rotate(matrix: List[List[int]]) -> None:
    l = 0
    r = len(matrix) -1
    while l < r:
        matrix[l], matrix[r] = matrix[r], matrix[l]
        l += 1
        r -= 1
    
    for i in range(len(matrix)):
        for j in range(i):
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]


def setZeroes(matrix: List[List[int]]) -> None:
    """
    Do not return anything, modify matrix in-place instead.
    """
    rows, cols = len(matrix), len(matrix[0])
    for i in range(rows):
        if matrix[i][0] == 0:
            matrix[i][0] = 'F'
        for j in range(1,cols):
            if matrix[i][j] == 0:
                if matrix[i][0] != 'F':
                    matrix[i][0] = 0
                matrix[0][j] = 0
    Firstcol = False
    for i in range(1,rows):
        if matrix[i][0] == 'F':
            Firstcol = True
            matrix[i] = [0] * cols
        if matrix[i][0] == 0:
            matrix[i] = [0] * cols
    for j in range(1,cols):
        if matrix[0][j] == 0:
            for i in range(rows):
                matrix[i][j] = 0
    if matrix[0][0] == 0:
        matrix[0] = [0] * cols
    if matrix[0][0] == 'F':
        matrix[0] = [0] * cols
        Firstcol = True
    if Firstcol:
        for i in range(rows):
            matrix[i][0] = 0

def setZeroes(matrix: List[List[int]]) -> None:

    m = len(matrix)
    n = len(matrix[0])
    
    first_row_has_zero = False
    first_col_has_zero = False
    
    
    for row in range(m):
        for col in range(n):
            if matrix[row][col] == 0:
                if row == 0:
                    first_row_has_zero = True
                if col == 0:
                    first_col_has_zero = True
                matrix[row][0] = matrix[0][col] = 0

    
    for row in range(1, m):
        for col in range(1, n):
            matrix[row][col] = 0 if matrix[0][col] == 0 or matrix[row][0] == 0 else matrix[row][col]
    
    
    if first_row_has_zero:
        for col in range(n):
            matrix[0][col] = 0
    
    if first_col_has_zero:
        for row in range(m):
            matrix[row][0] = 0
                
def gameOfLife(board: List[List[int]]) -> None:
    new = [row.copy() for row in board]
    rows = len(board)
    columns = len(board[0])
    total = 0
    for i in range(rows):
        for j in range(columns):
            for k in range(-1,2):
                for l in range(-1,2):
                    total += board[i+k][j+l] if (0 <= i+k < rows and 0 <= j+l < columns) else 0
            if board[i][j] == 0 and total == 3:
                new[i][j] = 1
                total = 0
                continue
            total -= board[i][j]
            if total < 2:
                new[i][j] = 0
            if total > 3:
                new[i][j] = 0
            total = 0
    for i in range(rows):
        for j in range(columns):
            board[i][j] = new[i][j]

def gameOfLife(board: List[List[int]]) -> None:
    directions = [(1,0), (1,-1), (0,-1), (-1,-1), (-1,0), (-1,1), (0,1), (1,1)]
    for i in range(len(board)):
        for j in range(len(board[0])):
            live = 0                
            for x, y in directions: 
                if ( i + x < len(board) and i + x >= 0 ) and ( j + y < len(board[0]) and j + y >=0 ) and abs(board[i + x][j + y]) == 1:
                    live += 1
            if board[i][j] == 1 and (live < 2 or live > 3):     
                board[i][j] = -1
            if board[i][j] == 0 and live == 3:                  
                board[i][j] = 2
    for i in range(len(board)):
        for j in range(len(board[0])):
            board[i][j] = 1 if(board[i][j] > 0) else 0


def gameOfLife(board):
    directions = [(1,0), (1,-1), (0,-1), (-1,-1), (-1,0), (-1,1), (0,1), (1,1)]
    change = True
    iteration = 0
    while change:
        iteration += 1
        change = False
        for i in range(len(board)):
            for j in range(len(board[0])):
                live = 0                
                for x, y in directions: 
                    if ( i + x < len(board) and i + x >= 0 ) and ( j + y < len(board[0]) and j + y >=0 ) and abs(board[i + x][j + y]) == 1:
                        live += 1
                if board[i][j] == 1 and (live < 2 or live > 3):     
                    board[i][j] = -1
                    change = True
                if board[i][j] == 0 and live == 3:                  
                    board[i][j] = 2
                    change = True
        for i in range(len(board)):
            for j in range(len(board[0])):
                board[i][j] = 1 if(board[i][j] > 0) else 0
        if iteration % 100 == 0:
            print("After 100 iterations:")
            for row in board:
                print(row)
            decision = input("Would you like to continue? Press Enter to continue or type 'stop' to exit: ").strip().lower()
            if decision == "stop":
                print(f"Game stopped after {iteration} iterations.")
                return board
    return board

def canConstruct(ransomNote: str, magazine: str) -> bool:
    ransomletters = Counter(ransomNote)
    magazineletters = Counter(magazine)
    for i in ransomletters:
        if ransomletters[i] > magazineletters[i]:
            return False
    return True

def canConstruct(ransomNote: str, magazine: str) -> bool:
    st1, st2 = Counter(ransomNote), Counter(magazine)
    return st1 & st2 == st1

def isIsomorphic(s: str, t: str) -> bool:
    map1 = []
    map2 = []
    for idx in s:
        map1.append(s.index(idx))
    for idx in t:
        map2.append(t.index(idx))
    if map1 == map2:
        return True
    return False

def isIsomorphic(self, s: str, t: str) -> bool:
    zipped_set = set(zip(s, t))
    return len(zipped_set) == len(set(s)) == len(set(t))


def wordPattern(pattern: str, s: str) -> bool:
    listofwords = s.split()
    if len(listofwords) != len(pattern):
        return False
    words ={}
    for i in range(len(listofwords)):
        if listofwords[i] not in words:
            words[listofwords[i]] = pattern[i]
        else:
            if words[listofwords[i]] != pattern[i]:
                return False
    return len(set(words.values())) == len(words)

def wordPattern(pattern: str, s: str) -> bool:
    words = s.split()
    if len(words) != len(pattern):
        return False
    zipped_set = set(zip(pattern, words))
    return len(zipped_set) == len(set(pattern)) == len(set(s.split()))

def wordPattern(self, p: str, s: str) -> bool:
    words, w_to_p = s.split(' '), dict()

    if len(p) != len(words): return False
    if len(set(p)) != len(set(words)): return False 

    for i in range(len(words)):
        if words[i] not in w_to_p: 
            w_to_p[words[i]] = p[i]
        elif w_to_p[words[i]] != p[i]: 
            return False

    return True

def isAnagram(s: str, t: str) -> bool:
    return Counter(s) == Counter(t)

def isAnagram(s: str, t: str) -> bool:
    tracker = collections.defaultdict(int)
    for x in s: tracker[x] += 1
    for x in t: tracker[x] -= 1
    return all(x == 0 for x in tracker.values())

def isAnagram(s: str, t: str) -> bool:
    sorted_s = sorted(s)
    sorted_t = sorted(t)
    return sorted_s == sorted_t

def isAnagram(self, s: str, t: str) -> bool:
    flag = True
    if len(s) != len(t): 
        flag = False
    else:
        letters = "abcdefghijklmnopqrstuvwxyz"
        for letter in letters:
            if s.count(letter) != t.count(letter):
                flag = False
                break
    return flag

def groupAnagrams(strs: List[str]) -> List[List[str]]:
    dictionary = {''.join(sorted(strs[0])): 0}
    output = [[strs[0]]]
    pos = 1
    for i in range(1,len(strs)):
        if ''.join(sorted(strs[i])) in dictionary:
            output[dictionary[''.join(sorted(strs[i]))]].append(strs[i])
        else:
            output.append([strs[i]])
            dictionary[''.join(sorted(strs[i]))] = pos
            pos +=1
    return output

def groupAnagrams(strs: List[str]) -> List[List[str]]:
    anagram_map = defaultdict(list)
    
    for word in strs:
        sorted_word = ''.join(sorted(word))
        anagram_map[sorted_word].append(word)
    
    return list(anagram_map.values())

def twoSum(nums: List[int], target: int) -> List[int]:
    dic={}
    for key,val in enumerate(nums):
        if val in dic:
            return dic[val],key
        else:
            dic[target-val]=key

def twoSum(nums: List[int], target: int) -> List[int]:
    n = len(nums)
    for i in range(n - 1):
        for j in range(i + 1, n):
            if nums[i] + nums[j] == target:
                return [i, j]
            

def twoSum(nums: List[int], target: int) -> List[int]:
    numMap = {}
    n = len(nums)
    for i in range(n):
        complement = target - nums[i]
        if complement in numMap:
            return [numMap[complement], i]
        numMap[nums[i]] = i

def isHappy(n: int) -> bool:
    seen = set()
    output = 0
    while n not in seen:
        seen.add(n)
        string = str(n)
        for j in range(len(string)):
            output += int(string[j])**2
        if output == 1:
            return True
        n = output
        output = 0
    return False


class Solution:
    def isHappy(self, n: int) -> bool:
        slow = self.squared(n)
        fast = self.squared(self.squared(n))

        while slow!=fast and fast!=1:
            slow = self.squared(slow)
            fast = self.squared(self.squared(fast))
        return fast==1


    
    def squared(self, n):
        result = 0
        while n>0:
            last = n%10
            result += last * last
            n = n//10
        return result


def isHappy(n: int) -> bool:
    hset = set()
    while n != 1:
        if n in hset: return False
        hset.add(n)
        n = sum([int(i) ** 2 for i in str(n)])
    else:
        return True
    

def containsNearbyDuplicate(nums: List[int], k: int) -> bool:
    numMap = {}
    for i in range(len(nums)):
        if nums[i] in numMap:
            if i - numMap[nums[i]] <= k:
                return True 
        numMap[nums[i]] = i
    return False
            

def containsNearbyDuplicate(nums: List[int], k: int) -> bool:
    
    lookup = {}
    
    for i in range(len(nums)):
        
        
        if nums[i] in lookup and abs(lookup[nums[i]]-i) <= k:
            return True
        
        
        lookup[nums[i]] = i
    
    return False

def containsNearbyDuplicate(nums: List[int], k: int) -> bool:
    s = set()
    for i in range(len(nums)):
        if nums[i] not in s:
            s.add(nums[i])
        else:
            if abs(i - nums.index(nums[i])) <= k:
                return True
            nums[nums.index(nums[i])] = -math.inf
    return False

def containsNearbyDuplicate_default(nums, k):
    if len(nums) <= 1: return False 
    if len(set(nums)) == len(nums): return False
    d = defaultdict(list)
    for ix, v in enumerate(nums):
        d[v].append(ix)
    indices = [v for v in d.values() if len(v) > 1]
    for ind in indices:
        return True if any(abs(ind[i] - ind[i + 1]) <= k for i in range(len(ind) - 1)) else False

def longestConsecutive(self, nums: List[int]) -> int:
    if not nums:
        return 0
    maxlength = 1
    nums = set(nums)
    for num in nums:
        if num - 1 not in nums:  
            current_num = num
            current_length = 1

            while current_num + 1 in nums:
                current_num += 1
                current_length += 1

            maxlength = max(maxlength, current_length)
    return maxlength


def longestConsecutive(nums: List[int]) -> int:
    if not nums:
        return 0
    maxlength = 1
    nums = set(nums)
    while nums:
        i = nums.pop()
        count = 1
        j = i
        while j+1 in nums:
            nums.remove(j+1)
            count += 1
            j += 1
        j = i
        while j-1 in nums:
            nums.remove(j-1)
            count += 1
            j -= 1
        maxlength = max(count,maxlength)
    return maxlength


def longestConsecutive(nums: List[int]) -> int:
    longest = 0
    num_set = set(nums)

    for n in num_set:
        if (n-1) not in num_set:
            length = 1
            while (n+length) in num_set:
                length += 1
            longest = max(longest, length)
    
    return longest


def summaryRanges(self, nums: List[int]) -> List[str]:
    if not nums:
        return []
    output =[]
    start = nums[0]
    for i in range(len(nums)-1):
        if nums[i + 1] - 1 != nums[i]:
            if start == nums[i]:
                output.append(str(start))
            else:
                output.append(f"{start}->{nums[i]}")
            start = nums[i + 1]
    
    if start == nums[-1]:
        output.append(str(start))
    else:
        output.append(f"{start}->{nums[-1]}")
    return output
            
def summaryRanges(nums: List[int]) -> List[str]:
    result = []

    start , end = 0 , 0

    while start < len(nums) and end < len(nums):

        if (end + 1) < len(nums) and nums[end] + 1 == nums[end + 1]:
            end = end + 1
        else:

            if nums[start] == nums[end]:
                result.append(str(nums[start]))
                start = start + 1
                end = end + 1

            else:
                result.append(str(nums[start]) + '->' + str(nums[end]))
                end = end + 1
                start = end

    return result

def summaryRanges(self, nums: List[int]) -> List[str]:
    ranges = [] 
    for i, n in enumerate(nums):
        if ranges and ranges[-1][1] == n-1:
            ranges[-1][1] = n
        else:
            ranges.append([n, n])

    return [f'{x}->{y}' if x != y else f'{x}' for x, y in ranges]



def summaryRanges(self, nums: List[int]) -> List[str]:
    t = 0
    ans = []
    nums.append('#')
    for i in range(1, len(nums)):
        if nums[i] == '#' or nums[i] - nums[t] != i - t:
            if i - t > 1:
                ans.append(f"{nums[t]}->{nums[i-1]}")
            else:
                ans.append(str(nums[t]))
            t = i
    return ans

def merge(intervals: List[List[int]]) -> List[List[int]]:
    intervals.sort(key=lambda x: x[0])
    output = [intervals.pop(0)]
    j = 0
    while j < len(intervals):
        unique = True
        for k in range(len(output)):
            if intervals[j][0] <= output[k][1] and intervals[j][1] >= output[k][0]:
                if intervals[j][0] < output[k][0]:
                    output[k][0] = intervals[j][0]
                if intervals[j][1] > output[k][1]:
                    output[k][1] = intervals[j][1]
                intervals.pop(0)
                unique = False
                break
        if unique:
            output += [intervals.pop(0)]
    return output

def merge(intervals: List[List[int]]) -> List[List[int]]:
    intervals = sorted(intervals, key=lambda x: x [0])
    ans = []
    for interval in intervals:
        if not ans or ans[-1][1] < interval[0]:
            ans.append(interval)
        else:
            ans[-1][1] = max(ans[-1][1], interval[1])
    
    return ans


def insert(intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
    if not intervals:
        return [newInterval]
    remove = []
    for i in range(len(intervals)):
        if intervals[i][1] < newInterval[0]:
            continue
        if newInterval[1] < intervals[i][0]:
            intervals.insert(i,newInterval)
            break

        remove += [i]
        newInterval[0] = min(newInterval[0],intervals[i][0])
        newInterval[1] = max(newInterval[1],intervals[i][1])
    else:
        intervals.append(newInterval)

    for r in range(len(remove)-1,-1,-1):
        intervals.pop(remove[r])
    return intervals

def insert(intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
    if not intervals:
        return [newInterval]
    output = []
    append = True
    for i in range(len(intervals)):
        if intervals[i][1] < newInterval[0]:
            output.append(intervals[i])
            continue
        if newInterval[1] < intervals[i][0]:
            if append:   
                append = False
                output.append(newInterval)
            output.append(intervals[i])
            continue
        newInterval[0] = min(newInterval[0],intervals[i][0])
        newInterval[1] = max(newInterval[1],intervals[i][1])
    if append:
        output.append(newInterval)
    return output

def insert(intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
    output = []
    for interval in intervals:
        if interval[1] < newInterval[0]:
            output.append(interval)
        elif newInterval[1] < interval[0]: 
            output.append(newInterval)
            newInterval = interval
        else:
            newInterval[0] = min(newInterval[0],interval[0])
            newInterval[1] = max(newInterval[1],interval[1])
    output.append(newInterval)
    return output

def insert(intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
    
    
    START, END = 0, 1
    
    s, e = newInterval[START], newInterval[END]
    
    left, right = [], []
    
    for cur_interval in intervals:
        
        if cur_interval[END] < s:
            
            left += [ cur_interval ]
            
        elif cur_interval[START] > e:
            
            right += [ cur_interval ]
            
        else:
            
            
            s = min(s, cur_interval[START])
            e = max(e, cur_interval[END])
            
    return left + [ [s, e] ] + right    


def insert(intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
    heap, ans, = [], [] 
    for s, e in intervals + [newInterval]: 
        heapq.heappush(heap, (s, -1))
        heapq.heappush(heap, (e, 1))
    cur, s = 0, None            
    while heap:                            
        i, val = heapq.heappop(heap)       
        if s is None: s = i                
        cur += val                         
        if not cur:                        
            ans.append([s, i])             
            s = None                       
    return ans        


def findMinArrowShots(points: List[List[int]]) -> int:
    points.sort()
    arrows = 0
    check = False
    for point in points:
        if not check or check[1] < point[0]:
            check = point
            arrows +=1
        if check[1] > point[1]:
            check[1] = point[1]
    
    return arrows

def findMinArrowShots(self, points: List[List[int]]) -> int:
    points.sort(key = lambda x: x[0])
    arrows = 0
    check = False
    for point in points:
        if not check or check[1] < point[0]:
            check = point
            arrows +=1
        if check[1] > point[1]:
            check[1] = point[1]
    
    return arrows

def findMinArrowShots(points: List[List[int]]) -> int:
    points.sort(key = lambda x: x[1])      
                                            
    tally, bow = 1, points[0][1]
                                            
    for start, end in points:              
        if bow < start:                    
            bow = end                      
            tally += 1                     
                                            
    return tally 

def findMinArrowShots(points: List[List[int]]) -> int:
    points.sort(key = lambda x : x[0])
    stack = [  ]
    for sp, ep in points:
        if len(stack)>0 and stack[-1][1] >= sp:
            last_sp, last_ep = stack.pop()
            stack.append( [max(sp, last_sp), min(ep, last_ep)] )
        else:
            stack.append([sp, ep])
    return len(stack)

def isValid(s: str) -> bool:
    chars = {"(":")","{":"}","[":"]"}
    stack = []
    for char in s:
        if char in chars:
            stack += [chars[char]]
        elif not stack or stack.pop() != char:
            return False
    return False if stack else True

def isValid(s: str) -> bool:
    while len(s) > 0:
        l = len(s)
        s = s.replace('()','').replace('{}','').replace('[]','')
        if l==len(s): return False
    return True

def isValid(s):
    stack = [] 
    for c in s: 
        if c in '([{': 
            stack.append(c) 
        else: 
            if not stack or \
                (c == ')' and stack[-1] != '(') or \
                (c == '}' and stack[-1] != '{') or \
                (c == ']' and stack[-1] != '['):
                return False 
            stack.pop() 
    return not stack 
                        


def simplifyPath(path: str) -> str:
    path += "/"
    stack = []
    word = ''
    for char in path:
        if char == '/':
            if word == '..':
                if stack:
                    stack.pop()
            elif word and word !='.':
                stack.append(word)
            word = ''
            continue
        word += char
    return '/'+'/'.join(stack)

def simplifyPath(path: str) -> str:
    dirOrFiles = []
    path = path.split("/")
    for elem in path:
        if dirOrFiles and elem == "..":
            dirOrFiles.pop()
        elif elem not in [".", "", ".."]:
            dirOrFiles.append(elem)
            
    return "/" + "/".join(dirOrFiles)

class MinStack:

    def __init__(self):
        self.list = []
        self.min = []
        

    def push(self, val: int) -> None:
        if not self.list:
            self.min += [val]
        elif self.list and self.min[-1] > val:
            self.min +=[val]
        else:
            self.min += [self.min[-1]]
        self.list += [val]

        

    def pop(self) -> None:
        self.list.pop()
        self.min.pop()
        

    def top(self) -> int:
        return self.list[-1]
        

    def getMin(self) -> int:
        return self.min[-1]
class MinStack:

    def __init__(self):
        self.list = []
        self.min = []
        

    def push(self, val: int) -> None:
        self.list.append(val)
        self.min.append( val if not self.min else min(val, self.min[-1]) )

        

    def pop(self) -> None:
        self.list.pop()
        self.min.pop()
        

    def top(self) -> int:
        return self.list[-1]
        

    def getMin(self) -> int:
        return self.min[-1]
    

class MinStack:
    def __init__(self):
        """
        initialize your data structure here.
        """
        self.stack = []
        self.currentMin = float('inf')
        self.prevMins = []
        
    def push(self, x: int) -> None:
        self.stack.append(x)
        if x <= self.currentMin:
            self.prevMins.append(self.currentMin)
            self.currentMin = x

    def pop(self) -> None:
        if self.stack[-1] == self.currentMin:
            self.currentMin = self.prevMins.pop()
        self.stack.pop()

    def top(self) -> int:
        return self.stack[-1]

    def getMin(self) -> int:
        return self.currentMin
    

class Node:
    def __init__(self, val=None, mini=None, next=None):
        
        self.val = val
        self.minimum = mini
        self.next = next

class MinStack:

    def __init__(self):
        """
        initialize your data structure here.
        """
        self.head = None
        

    def push(self, x: int) -> None:
	
	
        if self.head is None:
            node = Node(x, x)
            self.head = node
        else:
		
            node = Node(x, min(x, self.head.minimum), self.head)
            self.head = node
        

    def pop(self) -> None:
	
        self.head = self.head.next
        

    def top(self) -> int:
	
        return self.head.val
        

    def getMin(self) -> int:
	
        return self.head.minimum

def evalRPN(tokens: List[str]) -> int:
    stack = []
    for figure in tokens:
        if figure == '+':
            stack[-1] = stack.pop() + stack[-1]
        elif figure == '*':
            stack[-1] = stack.pop() * stack[-1]
        elif figure == '/':
            x = stack.pop()
            stack[-1] = int(stack[-1] / x)
        elif figure == '-':
            stack[-1] = -stack.pop() + stack[-1]
        else:
            stack += [int(figure)]
    return stack[0]

def evalRPN(tokens: List[str]) -> int:
    op = {'+': lambda x, y: y + x, 
            '-': lambda x, y: y - x,
            '*': lambda x, y: y * x,
            '/': lambda x, y: int(y / x)}
    s = []
    for t in tokens:
        if t in op:
            t = op[t](s.pop(), s.pop())
        s.append(int(t))
    return s[0]


def calculate(s: str) -> int:
    output, curr, sign, stack = 0, 0, 1, []
    for char in s:
        if char.isdigit():
            curr = curr*10 + int(char)
        elif char in '+-':
            output += curr * sign
            curr = 0
            if char == '+':
                sign = 1
            else:
                sign = -1
        elif char =='(':
            stack.append(output)
            stack.append(sign)
            sign = 1
            output = 0
        elif char == ')':
            output += curr * sign
            output = stack.pop()*output + stack.pop()
            curr = 0
    return output + curr * sign

def calculate(s):
    def evaluate(i):
        res, digit, sign = 0, 0, 1
        
        while i < len(s):
            if s[i].isdigit():
                digit = digit * 10 + int(s[i])
            elif s[i] in '+-':
                res += digit * sign
                digit = 0
                sign = 1 if s[i] == '+' else -1
            elif s[i] == '(':
                subres, i = evaluate(i+1)
                res += sign * subres
            elif s[i] == ')':
                res += digit * sign
                return res, i
            i += 1

        return res + digit * sign
    
    return evaluate(0)

class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

def hasCycle(head: Optional[ListNode]) -> bool:
    if not head:
        return False
    slow = head
    fast = head.next

    while slow!=fast and fast:
        slow = slow.next
        fast = fast.next
        if fast:
            fast = fast.next
    return bool(fast)

def hasCycle(head: ListNode) -> bool:
    dictionary = {}
    while head:
        if head in dictionary: 
            return True
        else: 
            dictionary[head]= True
        head = head.next
    return False

def hasCycle(head: Optional[ListNode]) -> bool:
    slow = head
    while(slow):
        if slow.val == None:
            
            return True
        slow.val = None 
        slow = slow.next
    return False

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def addTwoNumbers(l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
    carry = 0
    output = []
    final = False
    while l1 or l2:
        value = carry
        carry = 0
        if l1:
            value += l1.val
            l1 = l1.next
        if l2:
            value += l2.val
            l2 = l2.next
        if value >= 10:
            value = value - 10
            carry = 1
        output += [value]
    if carry == 1:
        output += [1]
    for i in range(len(output)-1,-1,-1):
        if final:    
            final = ListNode(output[i],final)
        else:
            final = ListNode(output[i])


    return final

def addTwoNumbers(l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
    dummyHead = ListNode(0)
    tail = dummyHead
    carry = 0

    while l1 is not None or l2 is not None or carry != 0:

        digit1 = l1.val if l1 is not None else 0
        digit2 = l2.val if l2 is not None else 0

        sum = digit1 + digit2 + carry
        digit = sum % 10
        carry = sum // 10

        newNode = ListNode(digit)
        tail.next = newNode
        tail = newNode

        l1 = l1.next if l1 is not None else None
        l2 = l2.next if l2 is not None else None

    result = dummyHead.next
    dummyHead.next = None
    return result

def addTwoNumbers(l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
    dummyHead = ListNode(0)
    tail = dummyHead
    carry = 0

    while l1 is not None or l2 is not None or carry != 0:

        digit1 = l1.val if l1 is not None else 0
        digit2 = l2.val if l2 is not None else 0

        sum = digit1 + digit2 + carry
        digit = sum % 10
        carry = sum // 10

        tail.next = ListNode(digit)
        tail = tail.next
        
        l1 = l1.next if l1 is not None else None
        l2 = l2.next if l2 is not None else None

    result = dummyHead.next
    return result

def addTwoNumbers(l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
    res = dummy = ListNode()
    carry = 0
    while l1 or l2:
        v1, v2 = 0, 0
        if l1: v1, l1 = l1.val, l1.next
        if l2: v2, l2 = l2.val, l2.next
        
        val = carry + v1 + v2
        res.next = ListNode(val%10)
        res, carry = res.next, val//10
        
    if carry:
        res.next = ListNode(carry)
        
    return dummy.next

def mergeTwoLists(list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
    res = dummy = ListNode()
    while list1 and list2:
        if list1.val<list2.val:
            val = list1.val
            list1 = list1.next
        else:
            val = list2.val
            list2 = list2.next
        res.next = ListNode(val)
        res = res.next
    while list1:
        res.next = ListNode(list1.val)
        res = res.next
        list1 = list1.next
    while list2:
        res.next = ListNode(list2.val)
        res = res.next
        list2 = list2.next



    return dummy.next

def mergeTwoLists(list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
    cur = dummy = ListNode()
    while list1 and list2:               
        if list1.val < list2.val:
            cur.next = list1
            list1, cur = list1.next, list1
            print(cur.val)
        else:
            cur.next = list2
            list2, cur = list2.next, list2
            
    if list1 or list2:
        cur.next = list1 if list1 else list2
        
    return dummy.next

def mergeTwoLists(l1, l2):  
    if not l1 or not l2:
        return l1 or l2
    
    if l1.val <= l2.val: 
        l1.next = mergeTwoLists(l1.next, l2)
        return l1
    else: 
        l2.next = mergeTwoLists(l1, l2.next)
        return l2

class Node:
    def __init__(self, val, next, random):
        self.val = val
        self.next = next
        self.random = random 

def copyRandomList(head: 'Optional[Node]') -> 'Optional[Node]':
    start = head
    dummy = Node(1)
    Matcher = {None: None}
    while head:
        Matcher[head] = Node(head.val)
        head = head.next
    dummy.next = Matcher[start]
    while start:
        Matcher[start].next = Matcher[start.next] 
        Matcher[start].random = Matcher[start.random] 
        start = start.next
    return dummy.next

def copyRandomList(head: 'Optional[Node]') -> 'Optional[Node]':
    if not head:
        return None
    start = head
    while start:
        start.next = Node(start.val, start.next,start.random)
        start = start.next.next
    start = head.next

    while start and start.next:
        print(start.val)
        start.next = start.next.next
        if start.random:
            start.random = start.random.next 
        start = start.next
    if start.random:
        start.random = start.random.next 
    return head.next

def reverseBetween(head: Optional[ListNode], left: int, right: int) -> Optional[ListNode]:
    traverse = head
    Tracker = 1
    reverse = []
    while Tracker <= right:
        if Tracker >= left:
            reverse.append(traverse.val)
        Tracker +=1 
        traverse = traverse.next
    traverse = head
    Tracker = 1
    while Tracker <= right:
        if Tracker >= left:
            traverse.val = reverse.pop()
        Tracker +=1 
        traverse = traverse.next
    return head

def reverseBetween(head: Optional[ListNode], left: int, right: int) -> Optional[ListNode]:
    traverse = head
    Tracker = 1
    reverse = []
    halfway = (right-left+1)//2 + left - 1
    while Tracker <= right:
        if halfway >= Tracker and Tracker >= left:
            reverse.append(traverse)
        if Tracker > halfway and Tracker != (right+left)/2:
            left_node = reverse.pop()
            left_node.val, traverse.val = traverse.val, left_node.val
        Tracker +=1 
        traverse = traverse.next
    return head

def reverseBetween(head: Optional[ListNode], m: int, n: int) -> Optional[ListNode]:

    
    
    
    dummy = ListNode(0)
    dummy.next = head
    
    pre = dummy
    cur = dummy.next
    
    
    for i in range(1,m):
        cur = cur.next
        pre = pre.next
    
    
    
    for i in range(n-m):
        temp = cur.next
        cur.next = temp.next
        temp.next  = pre.next
        pre.next = temp
    
    return dummy.next

def reverseKGroup(head: Optional[ListNode], k: int) -> Optional[ListNode]:
    traverse = head
    start = traverse
    reverse = []
    while traverse:
        reverse.append(traverse.val)
        traverse = traverse.next
        if len(reverse) == k:     
            while len(reverse) > 0:
                start.val = reverse.pop()
                start = start.next
            start = traverse
    return head

def reverseKGroup(head: Optional[ListNode], k: int) -> Optional[ListNode]:
    dummy = ListNode(0)
    dummy.next = head
    traverse = head
    start = dummy
    second_traverse = head
    tracker = 0
    while traverse:
        traverse = traverse.next
        tracker +=1
        if tracker == k:     
            while tracker > 1:
                temp = second_traverse.next
                second_traverse.next = temp.next
                temp.next  = start.next
                start.next = temp
                tracker -=1
            tracker = 0
            start = second_traverse
            second_traverse = traverse
    return dummy.next

def reverseKGroup(head: ListNode, k: int) -> ListNode:        
    
    curr = head
    for _ in range(k):
        if not curr: return head
        curr = curr.next
            
            
    
    prev = None
    curr = head
    for _ in range(k):
        nxt = curr.next
        curr.next = prev
        prev = curr
        curr = nxt
    
    
    
    
    head.next = reverseKGroup(curr, k)
    return prev

def reverseKGroup(head: Optional[ListNode], k: int) -> Optional[ListNode]:
    dummy = ListNode()
    prev_group = dummy
    while head:
        j, group_end = 1, head 
        while j < k and head.next:
            head = head.next 
            j+=1
        group_start = head 
        next_group = head = head.next 

        if j != k:  
            break
        
        prev, cur = None, group_end
        while cur != next_group:
            cur.next, cur, prev = prev, cur.next, cur  

        prev_group.next = group_start
        prev_group = group_end
        group_end.next = next_group

    return dummy.next

def removeNthFromEnd(head: Optional[ListNode], n: int) -> Optional[ListNode]:
    dummy = ListNode(0,head)
    traverse = head
    counter = 1
    while traverse:
        traverse = traverse.next
        counter +=1
    traverse = dummy
    while counter > n+1:
        traverse = traverse.next
        counter -=1  
    traverse.next = traverse.next.next
    return dummy.next

def removeNthFromEnd(head: Optional[ListNode], n: int) -> Optional[ListNode]:
    dummy = ListNode(0,head)
    Second_pointer = dummy
    for i in range(n-1):
        head = head.next
    while head.next:
        head = head.next
        Second_pointer = Second_pointer.next
    Second_pointer.next =  Second_pointer.next.next
    return dummy.next

def deleteDuplicates(head: Optional[ListNode]) -> Optional[ListNode]:
    dummy = ListNode(0,head)
    Second_pointer = dummy
    while head and head.next:
        if head.val == head.next.val:
            while head.next and head.val == head.next.val:
                head = head.next
            Second_pointer.next = head.next
            head = head.next
            continue
        head = head.next
        Second_pointer = Second_pointer.next 
    return dummy.next


def rotateRight(head: Optional[ListNode], k: int) -> Optional[ListNode]:
    if not head:
        return head
    dummy = ListNode(0,head)
    Second_pointer = dummy
    length = 0
    while head:
        head = head.next
        length +=1
    head = dummy.next
    for i in range(k%length-1):
        head = head.next
    if k%length >0:
        while head.next:
            head = head.next
            Second_pointer = Second_pointer.next
        end = Second_pointer
        Second_pointer = Second_pointer.next
        end.next = None
        head.next = dummy.next
        dummy.next = Second_pointer
    return dummy.next

def rotateRight(head: ListNode, k: int) -> ListNode:
    
    if not head:
        return None
    
    lastElement = head
    length = 1
    
    while ( lastElement.next ):
        lastElement = lastElement.next
        length += 1

    
    
    k = k % length
        
    
    
    lastElement.next = head
    
    
    
    
    tempNode = head
    for _ in range( length - k - 1 ):
        tempNode = tempNode.next
    
    
    
    
    
    
    answer = tempNode.next
    tempNode.next = None
    
    return answer


def partition(head: Optional[ListNode], x: int) -> Optional[ListNode]:
    dummy_small = ListNode(0)
    small_node = dummy_small
    dummy_big = ListNode(0)
    big_node = dummy_big
    while head:
        current = head
        head = head.next
        if current.val < x:
            small_node.next = current
            small_node = small_node.next
        else:
            big_node.next = current
            big_node = big_node.next
            big_node.next = None
    small_node.next = dummy_big.next
    return dummy_small.next


def partition(head: Optional[ListNode], x: int) -> Optional[ListNode]:
    dummy_small = ListNode(0)
    small_node = dummy_small
    dummy_big = ListNode(0)
    big_node = dummy_big
    while head:
        if head.val < x:
            small_node.next = head
            small_node = head
        else:
            big_node.next = head
            big_node = head
        head = head.next
    
    big_node.next = None
    small_node.next = dummy_big.next
    return dummy_small.next


class ListNode:
    def __init__(self, key, value):
        self.key = key
        self.value = value
        self.prev = None
        self.next = None
class LRUCache:

    def __init__(self, capacity: int):
        self.dict = dict() 
        self.capacity = capacity
        self.head = ListNode(0, 0)
        self.tail = ListNode(-1, -1)
        self.head.next = self.tail
        self.tail.prev = self.head
        

    def get(self, key: int) -> int:
        if key in self.dict:
            self.dict[key].prev.next = self.dict[key].next
            self.dict[key].next.prev = self.dict[key].prev
            self.dict[key].next = self.head.next
            self.head.next.prev = self.dict[key]
            self.head.next = self.dict[key]
            self.dict[key].prev = self.head
            return self.dict[key].value
        else:
            return -1

    def put(self, key: int, value: int) -> None:
        if key in self.dict:
            self.dict[key].prev.next = self.dict[key].next
            self.dict[key].next.prev = self.dict[key].prev
            self.dict[key].next = self.head.next
            self.head.next.prev = self.dict[key]
            self.head.next = self.dict[key]
            self.dict[key].prev = self.head
            self.dict[key].value = value
        else:    
            if len(self.dict) >= self.capacity:
                if len(self.dict) == 0: return
                del self.dict[self.tail.prev.key]
                self.tail.prev.prev.next = self.tail
                self.tail.prev = self.tail.prev.prev
            self.dict[key] = ListNode(key,value)
            self.dict[key].prev = self.head
            self.dict[key].next = self.head.next
            self.head.next.prev = self.dict[key]
            self.head.next = self.dict[key]

class ListNode:
    def __init__(self, key, value):
        self.key = key
        self.value = value
        self.prev = None
        self.next = None

class LRUCache:

    def __init__(self, capacity: int):
        self.dic = dict() 
        self.capacity = capacity
        self.head = ListNode(0, 0)
        self.tail = ListNode(-1, -1)
        self.head.next = self.tail
        self.tail.prev = self.head

    def get(self, key: int) -> int:
        if key in self.dic:
            node = self.dic[key]
            self.removeFromList(node)
            self.insertIntoHead(node)
            return node.value
        else:
            return -1

    def put(self, key: int, value: int) -> None:
        if key in self.dic:             
            node = self.dic[key]
            self.removeFromList(node)
            self.insertIntoHead(node)
            node.value = value         
        else: 
            if len(self.dic) >= self.capacity:
                self.removeFromTail()
            node = ListNode(key,value)
            self.dic[key] = node
            self.insertIntoHead(node)
			
    def removeFromList(self, node):
        node.prev.next = node.next
        node.next.prev = node.prev
    
    def insertIntoHead(self, node):
        headNext = self.head.next 
        self.head.next = node 
        node.prev = self.head 
        node.next = headNext 
        headNext.prev = node
    
    def removeFromTail(self):
        if len(self.dic) == 0: return
        tail_node = self.tail.prev
        del self.dic[tail_node.key]
        self.removeFromList(tail_node)


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def maxDepth(root: Optional[TreeNode]) -> int:
    if not root:
        return 0
    check = [root]
    output = 0
    while check:
        nodes = check
        check = []
        for node in nodes:
            if node.left:
                check += [node.left]
            if node.right:
                check += [node.right]
        output += 1
    return output
        

def maxDepth(self, root: Optional[TreeNode]) -> int:
    if not root:
        return 0
    return max(self.maxDepth(root.left), self.maxDepth(root.right)) + 1

def maxDepth(root: Optional[TreeNode]) -> int:
    if not root:
        return 0
    worklist = deque([root])
    num_node_level = 1
    levels = 0
    while worklist:
        node = worklist.popleft()
        if node.left:
            worklist.append(node.left)
        if node.right:
            worklist.append(node.right)
        num_node_level -= 1
        if num_node_level == 0:
            levels += 1
            num_node_level = len(worklist)
            
    return levels

def isSameTree(self, p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
    if not p and not q:
        return True
    if not p or not q or p.val != q.val:
        return False
    return self.isSameTree(p.left,q.left) and self.isSameTree(p.right, q.right)

def isSameTree(p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
    worklist = deque([p,q])
    num_node_level = 1
    while worklist:
        node1 = worklist.popleft()
        node2 = worklist.popleft()
        if node1 and node2 and node1.val == node2.val:
            worklist.extend([node1.left,node2.left, node1.right, node2.right])
        elif not node1 and not node2:
            continue
        else:
            return False
        num_node_level -= 1
        if num_node_level == 0:
            num_node_level = len(worklist)/2
            
    return True

def isSameTree(p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
    stack = [(p, q)]
    while len(stack):
        first, second = stack.pop()
        if not first and not second: pass
        elif not first or not second: return False
        else:
            if first.val != second.val: return False
            stack.append((first.left, second.left))
            stack.append((first.right, second.right))
    return True

def invertTree(root: Optional[TreeNode]) -> Optional[TreeNode]:
    if not root:
        return root
    layer = deque([root])
    while layer:
        node = layer.popleft()
        node.left, node.right = node.right, node.left
        if node.left:
            layer.append(node.left)
        if node.right:
            layer.append(node.right)
        
    return root

def invertTree(self,root: Optional[TreeNode]) -> Optional[TreeNode]:
    if not root: 
        return root
    self.invertTree(root.left) 
    self.invertTree(root.right)  
    
    root.left, root.right = root.right, root.left
    return root 

def invertTree(root: Optional[TreeNode]) -> Optional[TreeNode]:
    q = collections.deque([root])
    while q:            
        if cur:= q.popleft():
            cur.right, cur.left = cur.left, cur.right
            q.extend([cur.left, cur.right])
    return root

def isSymmetric(root: Optional[TreeNode]) -> bool:
    nodes = collections.deque([root,root])
    while nodes:
        left_node = nodes.popleft().left
        right_node = nodes.popleft().right   
        if left_node and right_node and left_node.val == right_node.val:
            nodes.extend([left_node, right_node,right_node,left_node])
        elif left_node == right_node:
            continue
        else:
            return False
    return True
        

class Solution:
    def isSymmetric(self, root):
        
        if not root:
            return True
        
        return self.isSame(root.left, root.right)
    
    def isSame(self, leftroot, rightroot):
        
        if leftroot == None and rightroot == None:
            return True
        
        if leftroot == None or rightroot == None:
            return False
        
        if leftroot.val != rightroot.val:
            return False
        
        return self.isSame(leftroot.left, rightroot.right) and self.isSame(leftroot.right, rightroot.left)
    
class Solution:
    def isSymmetric(self, root):
        stack = []
        if root: stack.append([root.left, root.right])

        while(len(stack) > 0):
            left, right = stack.pop()
            
            if left and right:
                if left.val != right.val: return False
                stack.append([left.left, right.right])
                stack.append([right.left, left.right])
        
            elif left or right: return False
        
        return True
    

class Solution:
    def buildTree(self, preorder, inorder):
        idx_map = {}
        for index in range(0, len(inorder)): idx_map[inorder[index]] = index
        preorder = collections.deque(preorder)
        return self.helper(0, len(preorder) - 1, preorder, inorder, idx_map)

    def helper(self,left, right, preorder, inorder, idx_map):
        if left > right: return None
        root_val = preorder.popleft()
        root = TreeNode(root_val)
        
        root.left = self.helper(left,idx_map[root_val]-1, preorder,inorder, idx_map)
        root.right = self.helper(idx_map[root_val]+1, right, preorder,inorder, idx_map)
        return root

def buildTree(preorder, inorder):
    def build(stop):
        if inorder and inorder[-1] != stop:
            root = TreeNode(preorder.pop())
            root.left = build(root.val)
            inorder.pop()
            root.right = build(stop)
            return root
    preorder.reverse()
    inorder.reverse()
    return build(None)


class Solution:
    def buildTree(self, preorder, inorder):
        LeftMost = 0
        Traverse = 0
        def build(TurnRight):
            nonlocal Traverse
            nonlocal LeftMost
            print(f"LeftMost: {LeftMost}, Length of inorder: {len(inorder)}")
            if LeftMost != len(inorder) and inorder[LeftMost] != TurnRight:
                root = TreeNode(preorder[Traverse])
                Traverse +=1 
                root.left = build(root.val)
                LeftMost +=1 
                root.right = build(TurnRight)
                return root
            return None 
        return build('Dummy')


class Solution:
    def buildTree(self, inorder: List[int], postorder: List[int]) -> Optional[TreeNode]:
        idx_map = {}
        for index in range(0, len(inorder)): idx_map[inorder[index]] = index
        return self.helper(0,len(inorder)-1,inorder,postorder,idx_map)
    def helper(self,left, right, inorder, postorder, idx_map):
        if left > right:
            return None
        node = TreeNode(postorder.pop())
        node.right = self.helper(idx_map[node.val]+1,right,inorder,postorder,idx_map)
        node.left = self.helper(left, idx_map[node.val]-1,inorder,postorder,idx_map)
        return node

def buildTree(self, inorder: List[int], postorder: List[int]) -> Optional[TreeNode]:
    def build(stop):
        if inorder and inorder[-1] != stop:
            root = TreeNode(postorder.pop())
            root.right = build(root.val)
            inorder.pop()
            root.left = build(stop)
            return root
    return build(None)

class Node:
    def __init__(self, val: int = 0, left: 'Node' = None, right: 'Node' = None, next: 'Node' = None):
        self.val = val
        self.left = left
        self.right = right
        self.next = next

class Solution:
    def connect(self, root: 'Node') -> 'Node':
        layer = collections.deque([root,None])
        if not root:
            return None
        while layer:
            node = layer.popleft() 
            if not node:
                if layer:
                    layer.append(None)
                continue
            node.next = layer[0]
            if node.left:
                layer.append(node.left)
            if node.right:
                layer.append(node.right)

        return root
    

class Solution:
    def connect(self, root: 'Node') -> 'Node':
        if not root:
            return None
        q = deque()
        q.append(root)
        dummy=Node(-999) 
        while q:
            length=len(q) 
            
            prev=dummy
            for _ in range(length): 
                popped=q.popleft()
                if popped.left:
                    q.append(popped.left)
                    prev.next=popped.left
                    prev=prev.next
                if popped.right:
                    q.append(popped.right)
                    prev.next=popped.right
                    prev=prev.next                
                 
        return root
    

def connect(root: 'Node') -> 'Node':
    if not root:
        return None
    
    curr=root
    dummy=Node(-999)        
    head=root        

    while head:
        curr=head 
        prev=dummy 
        
        while curr:  
            if curr.left:
                prev.next=curr.left
                prev=prev.next
            if curr.right:
                prev.next=curr.right
                prev=prev.next                                                
            curr=curr.next
        head=dummy.next 
        dummy.next=None 
    return root


class Solution:
    def flatten(self, root: Optional[TreeNode]) -> None:
        if not root:
            return None
        head = root
        temp = root.right     
        root.right = self.flatten(root.left)
        root.left = None
        while root.right:
            root = root.right    
        root.right = self.flatten(temp)
        return head


def flatten(root: Optional[TreeNode]) -> None:
    pointer = root
    while pointer:
        while pointer.left:
            temp = pointer.right
            pointer.right = pointer.left
            pointer.left = None
            rightmost = pointer
            while rightmost.right:
                rightmost = rightmost.right
            rightmost.right = temp
            pointer = pointer.right
        pointer = pointer.right
    return root

class Solution:
    def flatten(self, root: TreeNode) -> None:
        cur = root
        while cur:
            if cur.left:
                prev = cur.left
                while prev.right:
                    prev = prev.right    
                
                prev.right = cur.right   
                cur.right = cur.left    
                cur.left = None   
            
            cur = cur.right

def flatten(root: TreeNode) -> None:
    def flatten(n: TreeNode) -> TreeNode:
        if n.left: 
            right, n.right, n.left = n.right, n.left, None 
            n = flatten(n.right)
            n.right = right 

        return flatten(n.right) if n.right else n 

    if root: flatten(root)
    return root


def flatten(self, root: TreeNode) -> None:
    """
    Do not return anything, modify root in-place instead.
    """
    self.previous_right = None
    def helper(root = root):
        if root:
            helper(root.right)
            helper(root.left)
            root.right, self.previous_right = self.previous_right, root
            root.left = None
    helper()

def hasPathSum(root: Optional[TreeNode], targetSum: int) -> bool:   
    layer = deque([root])
    while layer and root:
        node = layer.popleft()
        if not node.left and not node.right and node.val == targetSum:
            return True
        if node.left:
            node.left.val += node.val
            layer.extend([node.left])
        if node.right:
            node.right.val += node.val
            layer.extend([node.right])
    return False

def hasPathSum(root: Optional[TreeNode], targetSum: int) -> bool:   
    total = 0
    def helper(node,total):
        if not node:
            return False
        total += node.val
        if not node.left and not node.right and total ==targetSum:
            return True
        return helper(node.left,total) or helper(node.right,total)
    return helper(root,total)

def hasPathSum(self,root: TreeNode, sum: int) -> bool:
	if not root:
		return False
	if not root.left and not root.right and root.val == sum:
		return True
	return self.hasPathSum(root.left, sum - root.val) or self.hasPathSum(root.right, sum - root.val)

def hasPathSum(root: TreeNode, targetSum: int) -> bool:
    layer = [root]
    while layer and root:
        node = layer.pop()
        if not node.left and not node.right and node.val == targetSum:
            return True
        if node.left:
            node.left.val += node.val
            layer.extend([node.left])
        if node.right:
            node.right.val += node.val
            layer.extend([node.right])
    return False

def sumNumbers(root: Optional[TreeNode]) -> int:
    layer = deque([root])
    Total = 0
    while layer and root:
        node = layer.popleft()
        if not node.left and not node.right:
            Total += node.val
        if node.left:
            node.left.val += node.val*10
            layer.extend([node.left])
        if node.right:
            node.right.val += node.val*10
            layer.extend([node.right])
    return Total

def sumNumbers(root: Optional[TreeNode]) -> int:
    layer = [(root,root.val)]
    Total = 0
    while layer and root:
        node, node_value = layer.pop()
        if not node.left and not node.right:
            Total += node_value
        if node.left:
            left_value = node_value*10 + node.left.val
            layer.extend([(node.left,left_value)])
        if node.right:
            right_value = node_value*10 + node.right.val
            layer.extend([(node.right,right_value)])
    return Total

def sumNumbers(root: Optional[TreeNode]) -> int:
    total = 0
    def helper(node,total):
        if not node:
            return 0
        total = total*10 + node.val
        if not node.left and not node.right:
            return total
        return helper(node.left,total) + helper(node.right,total)
    return helper(root,total)

def sumNumbers(self, root: Optional[TreeNode]) -> int:
    
    tot_sum, cur, depth = 0, 0, 0
    
    
    while root:
        
        
        if root.left:
            
            
            pre, depth = root.left, 1
            
            
            while pre.right and pre.right != root:
                pre, depth = pre.right, depth + 1
            
            
            if not pre.right:
                
                pre.right = root
                
                cur = cur * 10 + root.val
                
                root = root.left
            else:
                
                pre.right = None
                
                if not pre.left: 
                    tot_sum += cur
                
                cur //= 10 ** depth
                
                root = root.right
        else:
            
            cur = cur * 10 + root.val
            
            if not root.right:
                tot_sum += cur
            
            root = root.right
    
    
    return tot_sum


def maxPathSum(self, root: Optional[TreeNode]) -> int:
    if root:
        self.output = root.val
    def helper(node):
        if not node:
            return 0
        left_value = helper(node.left)
        right_value = helper(node.right)
        best_solution = node.val
        if left_value > 0:   
            best_solution += left_value
        if right_value > 0: 
            best_solution += right_value
        self.output = max(self.output,best_solution)
        return max(node.val, node.val + max(left_value, right_value))
    potential = helper(root)
    self.output = max(self.output, potential)

    return self.output


def maxPathSum(self, root: Optional[TreeNode]) -> int:
    self.output = -math.inf
    def helper(node):
        if not node:
            return 0
        left_value = max(0, helper(node.left))
        right_value = max(0, helper(node.right))
        max_sum_through_current_node = node.val + left_value + right_value
        self.output = max(self.output,max_sum_through_current_node)
        return node.val + max(left_value, right_value)
    helper(root)

    return self.output


class Solution(object):
    def maxPathSum(self, root):
        self.res = float('-inf')
        self.helper(root)
        return self.res 
        
    def helper(self, root):
        if not root:
            return 0
        left, right = self.helper(root.left), self.helper(root.right)
        self.res = max(self.res, root.val + left + right)
        return max(root.val + max(left, right), 0)



class BSTIterator:

    def __init__(self, root: Optional[TreeNode]):
        self.inorder = deque([])
        while root:
            if root.left:
                pre = root.left
                while pre.right and pre.right != root:
                    pre = pre.right
                if not pre.right:
                    pre.right = root
                    root = root.left
                else:
                    self.inorder.extend([root.val])
                    pre.right = None
                    root = root.right
            else:
                self.inorder.extend([root.val])
                root = root.right


        

    def next(self) -> int:
        return self.inorder.popleft()

        

    def hasNext(self) -> bool:
        if len(self.inorder) > 0:
            return True
        return False



class BSTIterator:

    def __init__(self, root: Optional[TreeNode]):
        self.root = root

        

    def next(self) -> int:
        while self.root:
            if self.root.left:
                pre = self.root.left
                while pre.right and pre.right != self.root:
                    pre = pre.right
                if not pre.right:
                    pre.right = self.root
                    self.root = self.root.left
                else:
                    to_return = self.root.val
                    pre.right = None
                    self.root = self.root.right
                    return to_return
            else:
                to_return = self.root.val
                self.root = self.root.right
                return to_return

        

    def hasNext(self) -> bool:
        if self.root:
            return True
        return False
    


class BSTIterator:
    def __init__(self, root: Optional[TreeNode]):
        self.iter = self._inorder(root)
        self.nxt = next(self.iter, None)
    
    def _inorder(self, node: Optional[TreeNode]) -> Generator[int, None, None]:
        if node:
            yield from self._inorder(node.left)
            yield node.val
            yield from self._inorder(node.right)

    def next(self) -> int:
        res, self.nxt = self.nxt, next(self.iter, None)
        return res

    def hasNext(self) -> bool:
        return self.nxt is not None

class BSTIterator:
    def __init__(self, root: TreeNode):
        self.stack = []
        while root:
            self.stack.append(root)
            root = root.left
    
    def next(self) -> int:
        node = self.stack.pop()
        if node.right:
            current = node.right
            while current:
                self.stack.append(current)
                current = current.left
        return node.val
    
    def hasNext(self) -> bool:
        return len(self.stack) > 0

class Solution:
    def countNodes(self, root: Optional[TreeNode]) -> int:
        if not root:
            return 0
        self.found_height = False
        self.total = 0
        def helper(node,height):
            height += 1
            if not node.left and not self.found_height:
                self.found_height = height
            if not node.left and height != self.found_height:
                return False
            if height == self.found_height:
                self.total += 1
                return True
            if helper(node.left, height):
                if not node.right and height != self.found_height:
                    return False
                return(helper(node.right, height))
            return False
            
        helper(root,0)
        return 2**(self.found_height) -1 - (2**(self.found_height-1) - self.total)



class Solution:
    def countNodes(self, root: TreeNode) -> int:
        
        
        
        
        
        
        
        
        
        
        
        
        if not root:
            return 0
        
        def depthLeft(node):
            d = 0
            while node:
                d += 1
                node = node.left
            return d

        def depthRight(node):
            d = 0
            while node:
                d += 1
                node = node.right
            return d
        
        ld = depthLeft(root.left)
        rd = depthRight(root.right)
        
        if ld == rd:
            return 2**(ld + 1) - 1
        else:
            return 1 + self.countNodes(root.left) + self.countNodes(root.right)
        
class Solution:
  def countNodes(self, root: Optional[TreeNode]) -> int:
    if not root:
      return 0

    l = root
    r = root
    heightL = 0
    heightR = 0

    while l:
      heightL += 1
      l = l.left

    while r:
      heightR += 1
      r = r.right

    if heightL == heightR:  
      return pow(2, heightL) - 1
    return 1 + self.countNodes(root.left) + self.countNodes(root.right)
  

class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        answer = None
        def helper(root):
            nonlocal answer
            if not root:
                return 0 
            indicator = 0
            if root == p or root == q:
                indicator +=1
            indicator += helper(root.left)
            indicator += helper(root.right) 
            if indicator == 2 and not answer:
                answer = root

            return indicator
        helper(root)
        return answer

def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
    if root in (None, p, q): return root
    left, right = (self.lowestCommonAncestor(kid, p, q)
                for kid in (root.left, root.right))
    return root if left and right else left or right

def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
    if not root or root == p or root == q:
        return root

    l = self.lowestCommonAncestor(root.left, p, q)
    r = self.lowestCommonAncestor(root.right, p, q)

    if l and r:
        return root
    return l or r

def rightSideView(self, root: Optional[TreeNode]) -> List[int]:
    if not root:
        return []
    output = [root.val]
    nodes = deque([root,None])
    right_found = False
    while nodes:
        node = nodes.popleft()
        if not node:
            if nodes:
                right_found = False
                nodes.extend([None])
            continue
        if node.right:
            nodes.extend([node.right])
            if not right_found:
                output += [node.right.val]
                right_found = True
        if node.left:
            nodes.extend([node.left])
            if not right_found:
                output += [node.left.val]
                right_found = True
        
    return output

def rightSideView(self, root: Optional[TreeNode]) -> List[int]:
    if not root:
        return []
    
    output = [root.val]
    nodes = deque([root, None])
    rightmost_node_found = False
    
    while nodes:
        node = nodes.popleft()
        
        if node is None:
            if nodes:
                nodes.append(None)
                rightmost_node_found = False
            continue
        
        for child in [node.right, node.left]:  
            if child:
                nodes.append(child)
                if not rightmost_node_found:
                    output.append(child.val)
                    rightmost_node_found = True

    return output


def rightSideView(self, root: Optional[TreeNode]) -> List[int]:
    
    def solve(root, lvl):
        if root:
            if len(res)==lvl:
                res.append(root.val)
            solve(root.right, lvl + 1)
            solve(root.left, lvl + 1)
        return 

    res = []
    solve(root,0)
    return res


class Solution:
    def rightSideView(self, root: TreeNode) -> List[int]:
        if not root:
            return []
        result = []
        queue = collections.deque()
        queue.append(root)
        while queue:
            level_len = len(queue)
            for i in range(level_len):
                node = queue.popleft()
                if i == level_len - 1:
                    result.append(node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
        return result



def averageOfLevels(self, root: Optional[TreeNode]) -> List[float]:
    result = []
    queue = deque([root])
    while queue:
        total = 0
        level_len = len(queue)
        for i in range(level_len):
            node = queue.popleft()
            total += node.val
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        result += [total/level_len]
    return result


def averageOfLevels(self, root: TreeNode) -> List[float]:
	lvlcnt = defaultdict(int)
	lvlsum = defaultdict(int)

	def dfs(node=root, level=0):
		if not node: return
		lvlcnt[level] += 1
		lvlsum[level] += node.val
		dfs(node.left, level+1)
		dfs(node.right, level+1)
		
	dfs()
	return [lvlsum[i] / lvlcnt[i] for i in range(len(lvlcnt))]

class Solution:
    def averageOfLevels(self, root: TreeNode) -> List[float]:
        
        if not root:
            
            
            return []
        
        traversal_q = [root]
        
        average = []
        
        while traversal_q:
            
            
            cur_avg = sum( (node.val for node in traversal_q if node) ) / len(traversal_q)
            
            
            average.append( cur_avg )
            
            
            next_level_q = [ child for node in traversal_q for child in (node.left, node.right) if child ]
            
            
            traversal_q = next_level_q
            
        return average
    
def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:              
    if not root:
        return []
    result = []
    queue = deque([root])
    while queue:
        row = []
        level_len = len(queue)
        for i in range(level_len):
            node = queue.popleft()
            row.append(node.val)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        result += [row]
    return result

def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:              
    rows = defaultdict(list)

    def dfs(node=root, level=0):
        if not node: return
        rows[level] += [node.val]
        dfs(node.left, level+1)
        dfs(node.right, level+1)
        
    dfs()
    return list(rows.values())


def zigzagLevelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
    rows = defaultdict(deque)

    def dfs(node=root, level=0):
        if not node: return
        if level%2 == 1:
            rows[level].appendleft(node.val)
        else:
            rows[level].append(node.val)
        dfs(node.left, level+1)
        dfs(node.right, level+1)
        
    dfs()
    return list(rows.values())

def zigzagLevelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
    if not root:
        return []
    result = []
    queue = deque([root])
    while queue:
        row = []
        level_len = len(queue)
        for i in range(level_len):
            node = queue.popleft()
            row.append(node.val)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        if len(result)%2 == 1:
            row = row[::-1]
        result += [row]
    return result


def zigzagLevelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
    if not root: return []
    queue = collections.deque([root])
    res = []
    even_level = False
    while queue:
        n = len(queue)
        level = [0] * n 
        for i in range(n):
            node = queue.popleft()
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
            if even_level:
                level[n-1-i] = node.val
            else:
                level[i] = node.val
        res.append(level)
        even_level = not even_level

    return res


def zigzagLevelOrder(self, root: TreeNode) -> List[List[int]]:
	if not root: return []
	queue = collections.deque([root])
	res = []
	even_level = False
	while queue:
		n = len(queue)
		level = []
		for i in range(n):
			if even_level:
				
				node = queue.pop()
				
				
				if node.right: queue.appendleft(node.right)
				if node.left: queue.appendleft(node.left)
			else:
				
				node = queue.popleft()
				
				if node.left: queue.append(node.left)
				if node.right: queue.append(node.right)
			level.append(node.val)
		res.append(level)
		even_level = not even_level
	return res


def zigzagLevelOrder(self, root: TreeNode) -> List[List[int]]:
    if not root: return []
    res = []
    queue = collections.deque([root])
    even_level = False
    while queue:
        n = len(queue)
        level = collections.deque()
        for _ in range(n):
            node = queue.popleft()
            if even_level:
                level.appendleft(node.val)
            else:
                level.append(node.val)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        res.append(list(level))
        even_level = not even_level
    return res

def getMinimumDifference(self, root: Optional[TreeNode]) -> int:
    def _inorder(node: Optional[TreeNode]) -> Generator[int, None, None]:
        if node:
            yield from _inorder(node.left)
            yield node.val
            yield from _inorder(node.right)
    minimum = math.inf
    generator = _inorder(root)
    current = next(generator,None)
    next_node = next(generator,None)
    while next_node:
        minimum = min(minimum,abs(current-next_node))
        current = next_node
        next_node = next(generator,None)
    return minimum


def getMinimumDifference(self, root: Optional[TreeNode]) -> int:
    cur, stack, minDiff, prev = root, [], 10**5, -10**5
    
    while stack or cur:
        while cur:
            stack.append(cur)
            cur = cur.left
        node = stack.pop()
        minDiff = min(minDiff, node.val - prev)
        prev = node.val
        cur = node.right
    
    return minDiff

def getMinimumDifference(self,root):
    def recurse(less, more, node):
        if not node:
            return float('inf')
        else:
            return min(
                node.val - less,
                more - node.val,
                recurse(less, node.val, node.left),
                recurse(node.val, more, node.right)
            )
    return recurse(float('-inf'), float('inf'), root)


def kthSmallest_dfs_early_stopping(self, root, k):
	res = []
	def _inorder(node):
		if not node: return
		_inorder(node.left)
		if len(res) == k:
			return
		res.append(node.val)
		_inorder(node.right)
	_inorder(root)
	return res[-1]







class Solution:
    def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:
        def _inorder(node: Optional[TreeNode]) -> Generator[int, None, None]:
            if node:
                yield from _inorder(node.left)
                yield node.val
                yield from _inorder(node.right)
        generator = _inorder(root)
        for i in range(k):
            output = next(generator)
        return output
class Solution:
    def kthSmallest(self, root: TreeNode, k: int) -> int:
        
        stack = []
        
        current = root
        
        
        while True:
            
            while current is not None:
                stack.append(current)
                current = current.left
            
            
            if not stack:
                break
                
            
            node = stack.pop()
            k -= 1
            
            
            if k == 0:
                return node.val
            
            
            current = node.right        

def isValidBST(self, root: Optional[TreeNode]) -> bool:
    def _inorder(node: Optional[TreeNode]) -> Generator[int, None, None]:
        if node:
            yield from _inorder(node.left)
            yield node.val
            yield from _inorder(node.right)
    generator = _inorder(root)
    output = next(generator,None)
    while output != None:
        check = output
        output = next(generator,None)
        if output != None and check >= output:
            return False
    return True
    
def isValidBST(self, root: Optional[TreeNode]) -> bool:
    prev = float('-inf')
    def inorder(node):
        nonlocal prev
        if not node:
            return True
        if not (inorder(node.left) and prev < node.val):
            return False
        prev = node.val
        return inorder(node.right)
    return inorder(root)

class Solution:
    def isValidBST(self, root: TreeNode) -> bool:
        
        
        INF = sys.maxsize
        
        def helper(node, lower, upper):
            
            if not node:
				
                return True
            
            if lower < node.val < upper:
				
                return helper(node.left, lower, node.val) and helper(node.right, node.val, upper)
            
            else:
				
                return False
            
        
        
        return helper( node=root, lower=-INF, upper=INF )


class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        rows = len(grid)
        columns = len(grid[0])
        total = 0
        for i in range(rows):
            for j in range(columns):
                if grid[i][j] =="1":
                    total += 1
                    coords = [[i,j]]
                    while coords:
                        x,y = coords.pop()
                        grid[x][y] = 2
                        for k in range(-1,2):
                            current = grid[x+k][y] if (0 <= x+k < rows) else 0
                            if current == "1":
                                coords += [[x+k,y]]
                        for l in range(-1,2):
                            current = grid[x][y+l] if (0 <= y+l < columns) else 0
                            if current == "1":
                                coords += [[x,y+l]]
        return total

from collections import deque
class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        """
        :type grid: List[List[str]]
        :rtype: int
        """
        if not grid:
            return 0   
        count = 0
        check = [[False for _ in range(len(grid[0]))] for _ in range(len(grid))]
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] =='1' and check[i][j]== False:
                    count += 1
                    self.search(grid,check,i,j)
        return count       
    def search(self,grid,check,i,j):
        qu = deque([(i,j)])
        while qu:
            i, j = qu.popleft()
            if 0<=i<len(grid) and 0<=j<len(grid[0]) and grid[i][j]=='1' and check[i][j]==False:
                check[i][j] = True
                qu.extend([(i-1,j),(i+1,j),(i,j-1),(i,j+1)])

from collections import deque
class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        count = 0
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == '1':
                    
                    grid[i][j] = '0'
                    self.helper(grid,i,j)
                    count += 1
        return count
    
    def helper(self,grid,i,j):
        queue = deque([(i,j)])
        while queue:
            I,J = queue.popleft()
            for i,j in [I+1,J],[I,J+1],[I-1,J],[I,J-1]:
                if 0 <= i < len(grid) and 0 <= j < len(grid[0]) and grid[i][j] == '1':
                    
                    grid[i][j] = '0'
                    queue.append((i,j))

class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        count = 0
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == '1':
                    print(i,j)
                    self.dfs(grid,i,j)
                    count  += 1
        
        return count
    
    def dfs(self,grid,i,j):
        grid[i][j] = 0
        for dr,dc in (1,0), (-1,0), (0,-1), (0,1):
            r = i + dr
            c = j + dc
            if 0 <= r < len(grid) and 0 <= c < len(grid[0]) and grid[r][c]=='1':
                self.dfs(grid,r,c)


def solve(self, board: List[List[str]]) -> None:
    for i in [0,len(board)-1]:
        for j in range(len(board[0])):
            if board[i][j] == 'O':
                board[i][j] = 'S'
                self.helper(board,i,j)
    for i in range(1,len(board)-1):
        for j in [0,len(board[0])-1]:
            if board[i][j] == 'O':
                board[i][j] = 'S'
                self.helper(board,i,j)
    for i in range(len(board)):
        for j in range(len(board[0])):
            if board[i][j] == 'O':
                board[i][j] = 'X'
            if board[i][j] == 'S':
                board[i][j] = 'O'

def helper(self,board,i,j):
    queue = deque([(i,j)])
    while queue:
        I,J = queue.popleft()
        for i,j in [I+1,J],[I,J+1],[I-1,J],[I,J-1]:
            if 0 <= i < len(board) and 0 <= j < len(board[0]) and board[i][j] == 'O':
                board[i][j] = 'S'
                queue.append((i,j))

    from collections import deque

class Solution:
    def solve(self, board: List[List[str]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        
        o = "O"
        
        n = len(board) 
        m = len(board[0])

        Q = deque()
        
        for i in range(n):
            if board[i][0] == o:
                Q.append((i,0))
            if board[i][m-1] == o:
                Q.append((i, m-1))
                
        for j in range(m):
            if board[0][j] == o:
                Q.append((0,j))
            if board[n-1][j] == o:
                Q.append((n-1, j))
                
        def inBounds(i,j):
            return (0 <= i < n) and (0 <= j < m)
                
        while Q:
            i,j = Q.popleft()
            board[i][j] = "#"
            
            for ii, jj in [(i+1, j), (i-1, j), (i, j+1), (i, j-1)]:
                if not inBounds(ii, jj):
                    continue
                if board[ii][jj] != o:
                    continue
                Q.append((ii,jj))
                board[ii][jj] = '#'
            
        for i in range(n):
            for j in range(m):
                if board[i][j] == o:
                    board[i][j] = 'X'
                elif board[i][j] == '#':
                    board[i][j] = o


class Solution:
    def solve(self, board: List[List[str]]) -> None:
        for i in [0,len(board)-1]:
            for j in range(len(board[0])):
                if board[i][j] == 'O':
                    self.helper(board,i,j)
        for i in range(1,len(board)-1):
            for j in [0,len(board[0])-1]:
                if board[i][j] == 'O':
                    self.helper(board,i,j)
        for i in range(len(board)):
            for j in range(len(board[0])):
                if board[i][j] == 'O':
                    board[i][j] = 'X'
                if board[i][j] == 'S':
                    board[i][j] = 'O'

    def helper(self,board,i,j):
        for dr,dc in (1,0), (-1,0), (0,-1), (0,1):
            board[i][j] = 'S'
            r = i + dr
            c = j + dc
            if 0 <= r < len(board) and 0 <= c < len(board[0]) and board[r][c]=='O':
                self.helper(board,r,c)

"""

class Node:
    def __init__(self, val = 0, neighbors = None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []
"""

from typing import Optional
class Solution:
    def cloneGraph(self, node: Optional['Node']) -> Optional['Node']:
        if not node:
            return None
        nodes = deque([node])
        list_of_nodes = []
        dummy = Node(1)
        Matcher = {None: None}
        while nodes:
            current = nodes.popleft()
            if current in Matcher:
                continue
            Matcher[current] = Node(current.val)
            list_of_nodes += [current]
            for neighbor in current.neighbors:
                    if neighbor and not neighbor in Matcher:
                        nodes.extend([neighbor])
        for current in list_of_nodes:
            for neighbor in current.neighbors:
                Matcher[current].neighbors += [Matcher[neighbor]]
        return Matcher[list_of_nodes[0]]
    

class Solution:
    def cloneGraph(self, node: Optional['Node']) -> Optional['Node']:
        if not node:
            return None
        nodes = deque([node])
        Matcher = {node: Node(node.val)}
        while nodes:
            current = nodes.popleft()
            for neighbor in current.neighbors:
                    if neighbor not in Matcher:
                        Matcher[neighbor] = Node(neighbor.val)
                        nodes.extend([neighbor])
                    Matcher[current].neighbors += [Matcher[neighbor]]
        return Matcher[node]

class Solution:
    def cloneGraph(self, node: Optional['Node']) -> Optional['Node']:
        if not node:
            return None
        nodes = deque([node])
        Matcher = {node.val: Node(node.val)}
        while nodes:
            current = nodes.popleft()
            for neighbor in current.neighbors:
                    if neighbor.val not in Matcher:
                        Matcher[neighbor.val] = Node(neighbor.val)
                        nodes.extend([neighbor])
                    Matcher[current.val].neighbors += [Matcher[neighbor.val]]
        return Matcher[node.val]


class Solution:
    
    def helper(self, node, visited):
        if node is None:
            return None
        
        newNode = Node(node.val)
        visited[node.val] = newNode
        
        for adjNode in node.neighbors:
            if adjNode.val not in visited:
                newNode.neighbors.append(self.helper(adjNode, visited))
            else:
                newNode.neighbors.append(visited[adjNode.val])
        
        return newNode
    
    def cloneGraph(self, node: 'Node') -> 'Node':
        return self.helper(node, {})


class Solution:
    def calcEquation(self, equations: List[List[str]], values: List[float], queries: List[List[str]]) -> List[float]:
        output = []
        equations_graph = defaultdict(dict)
        for i in range(len(equations)):
            a, b = equations[i]
            equations_graph[a][b] = values[i]
            equations_graph[b][a] = 1/values[i]
        for query in queries:
            a, b = query
            if a not in equations_graph or b not in equations_graph:
                output += [-1.0]
                continue
            output += [self.dfs(a,b,equations_graph,1.0,set())]
        return output

    def dfs(self,a,b,equations_graph,weight,visit):
        if a == b:
            return weight
        visit.add(a)
        for next_node, ratio in equations_graph[a].items():
            if next_node not in visit:
                result = self.dfs(next_node,b,equations_graph,weight*ratio,visit)
                if result != -1.0:
                    return result
        return -1.0

from typing import List

class Solution:
    def dfs(self, node: str, dest: str, gr: dict, vis: set, ans: List[float], temp: float) -> None:
        if node in vis:
            return

        vis.add(node)
        if node == dest:
            ans[0] = temp
            return

        for ne, val in gr[node].items():
            self.dfs(ne, dest, gr, vis, ans, temp * val)

    def buildGraph(self, equations: List[List[str]], values: List[float]) -> dict:
        gr = {}

        for i in range(len(equations)):
            dividend, divisor = equations[i]
            value = values[i]

            if dividend not in gr:
                gr[dividend] = {}
            if divisor not in gr:
                gr[divisor] = {}

            gr[dividend][divisor] = value
            gr[divisor][dividend] = 1.0 / value

        return gr

    def calcEquation(self, equations: List[List[str]], values: List[float], queries: List[List[str]]) -> List[float]:
        gr = self.buildGraph(equations, values)
        finalAns = []

        for query in queries:
            dividend, divisor = query

            if dividend not in gr or divisor not in gr:
                finalAns.append(-1.0)
            else:
                vis = set()
                ans = [-1.0]
                temp = 1.0
                self.dfs(dividend, divisor, gr, vis, ans, temp)
                finalAns.append(ans[0])

        return finalAns

from collections import defaultdict, deque

class Solution:
    def calcEquation(self, equations, values, queries):
        graph = self.buildGraph(equations, values)
        results = []
        
        for dividend, divisor in queries:
            if dividend not in graph or divisor not in graph:
                results.append(-1.0)
            else:
                result = self.bfs(dividend, divisor, graph)
                results.append(result)
        
        return results
    
    def buildGraph(self, equations, values):
        graph = defaultdict(dict)
        
        for (dividend, divisor), value in zip(equations, values):
            graph[dividend][divisor] = value
            graph[divisor][dividend] = 1.0 / value
        
        return graph
    
    def bfs(self, start, end, graph):
        queue = deque([(start, 1.0)])
        visited = set()
        
        while queue:
            node, value = queue.popleft()
            
            if node == end:
                return value
            
            visited.add(node)
            
            for neighbor, weight in graph[node].items():
                if neighbor not in visited:
                    queue.append((neighbor, value * weight))
        
        return -1.0


class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        graph = defaultdict(list)
        for courses in prerequisites:
            graph[courses[0]] += [courses[1]]
        visited = set()
        for course in graph:
            if course not in visited and not self.dfs(course,graph[course],graph,set(),visited):
                return False
        return True
            
    def dfs(self,course,pres,graph,visit,visited):
        if not pres or course in visited:
            return True
        visited.add(course)
        visit.add(course)
        for pre in pres:
            if pre in visit:
                return False
            if pre in graph and not self.dfs(pre,graph[pre],graph,visit,visited):
                return False
        visit.remove(course)
        return True


class Solution:
    def buildAdjacencyList(self, n, edgesList):
            adjList = [[] for _ in range(n)]
            
            
            for c1, c2 in edgesList:
                adjList[c2].append(c1)
            return adjList
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        
        adjList = self.buildAdjacencyList(numCourses, prerequisites)

        
        
        
        
        
        state = [0] * numCourses

        def hasCycle(v):
            if state[v] == 1:
                
                return False
            if state[v] == -1:
                
                return True

            
            state[v] = -1

            for i in adjList[v]:
                if hasCycle(i):
                    return True

            state[v] = 1
            return False

        
        for v in range(numCourses):
            if hasCycle(v):
                return False

        return True

class Solution:
    def buildAdjacencyList(self, n, edgesList):
            adjList = [[] for _ in range(n)]
            
            
            for c1, c2 in edgesList:
                adjList[c2].append(c1)
            return adjList
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
                
                adjList = self.buildAdjacencyList(numCourses, prerequisites)
                visited = set()
        
                def hasCycle(v, stack):
                    if v in visited:
                        if v in stack:
                            
                            return True
                        
                        return False
        
                    
                    visited.add(v)
                    
                    stack.append(v)
        
                    for i in adjList[v]:
                        if hasCycle(i, stack):
                            return True
        
                    
                    stack.pop()
                    return False
        
                
                for v in range(numCourses):
                    if hasCycle(v, []):
                        return False
        
                return True

class Solution:
    def buildAdjacencyList(self, n, edgesList):
            adjList = [[] for _ in range(n)]
            
            
            for c1, c2 in edgesList:
                adjList[c2].append(c1)
            return adjList
    def topoBFS(self, numNodes, edgesList):
            
            
            adjList = self.buildAdjacencyList(numNodes, edgesList)
    
            
            inDegrees = [0] * numNodes
            for v1, v2 in edgesList:
                
                inDegrees[v1] += 1
    
            
            
            
            
            queue = deque([])
            for v in range(numNodes):
                if inDegrees[v] == 0:
                    queue.append(v)
    
            
            count = 0
            
            topoOrder = []
    
            while queue:
                
                
                
                v = queue.popleft()
                
                topoOrder.append(v)
    
                
                count += 1
    
                
                for des in adjList[v]:
                    inDegrees[des] -= 1
                    
                    if inDegrees[des] == 0:
                        queue.append(des)
    
            if count != numNodes:
                return None  
            else:
                return topoOrder
    
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        return True if self.topoBFS(numCourses, prerequisites) else False


class Solution:
    def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
        Required = [[] for _ in range(numCourses)]
        inDegrees = [0] * numCourses
        for c1, c2 in prerequisites:
            Required[c2].append(c1)
            inDegrees[c1] += 1

        queue = deque([])

        for i in range(numCourses):
            if inDegrees[i] == 0:
                queue.append(i)
        count = 0
        topoOrder = []
    
        while queue:
            course = queue.popleft()
            count += 1
            topoOrder.append(course)
            for i in Required[course]:
                inDegrees[i] -= 1
                if inDegrees[i] == 0:
                    queue.append(i)
            
        if count != numCourses:
            return []
        else:
            return topoOrder
        

class Solution:
    def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
        
		
        preq = {i:set() for i in range(numCourses)}
		
        graph = collections.defaultdict(set)
        for i,j in prerequisites:
		    
            preq[i].add(j)
			
            graph[j].add(i)
        
        q = collections.deque([])
		
        for k, v in preq.items():
            if len(v) == 0:
                q.append(k)
		
        taken = []
        while q:
            course = q.popleft()
            taken.append(course)
			
            if len(taken) == numCourses:
                return taken
			
            for cor in graph[course]:
			    
                preq[cor].remove(course)
				
                if not preq[cor]:
                    q.append(cor)
		
        return []

class Solution:

    def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
            adj=[[] for i in range(numCourses)]
            for u,v in prerequisites:
                adj[v].append(u)
            visited=[0]*numCourses
            indeg=[0]*numCourses
            res=[]
            q=deque()
            for i in range(numCourses):
                for j in adj[i]:
                    indeg[j]+=1
            for i in range(numCourses):
                if indeg[i]==0:
                    q.append(i)
            while q:
                u=q.popleft()
                res.append(u)
                for i in adj[u]:
                    if indeg[i]!=0:
                        indeg[i]-=1
                    if indeg[i]==0:
                        q.append(i)
            if len(res)!=numCourses:
                return []
            return res
    

class Solution:
    def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
            def dfs(sv,visited):
                if visited[sv]==-1:
                    return False
                if visited[sv]==1:
                    return True
                visited[sv]=-1 
                for u in adj[sv]:
                    if not dfs(u,visited):
                        return False 
                res.append(sv) 
                visited[sv]=1  
                return True
            
            adj=[[] for i in range(numCourses)]
            res=[]
            for u,v in prerequisites:
                adj[v].append(u)
            visited=[0]*numCourses
            for i in range(numCourses):
                if not dfs(i,visited):
                    
                    return []
            return res[::-1]

class Solution:
    def snakesAndLadders(self, board: List[List[int]]) -> int:
        row_length= len(board)
        end = row_length*row_length
        positions = deque([(1,0)])
        visited = set()
        while positions:
            pos, roll = positions.popleft()
            roll +=1
            if pos in visited:
                continue
            visited.add(pos)
            for i in range(1,7):
                if (next_pos := i+pos) >= end:
                    return roll
                row = ((next_pos-1) // row_length) + 1
                col = ((next_pos-1) % row_length)
                if row % 2 == 0:
                    curr = board[row_length - row][row_length-col-1]
                else:
                    curr = board[row_length - row][col]
                if curr == end:
                    return roll
                if curr == -1:
                    positions.append((next_pos, roll))
                else:
                    positions.append((curr, roll))
        return -1

class Solution:
    def snakesAndLadders(self, board: List[List[int]]) -> int:
        n = len(board)
        def label_to_position(label):
            r, c = divmod(label-1, n)
            if r % 2 == 0:
                return n-1-r, c
            else:
                return n-1-r, n-1-c
            
        seen = set()
        queue = collections.deque()
        queue.append((1, 0))
        while queue:
            label, step = queue.popleft()
            r, c = label_to_position(label)
            if board[r][c] != -1:
                label = board[r][c]
            if label == n*n:
                return step
            for x in range(1, 7):
                new_label = label + x
                if new_label <= n*n and new_label not in seen:
                    seen.add(new_label)
                    queue.append((new_label, step+1))
        return -1
    

class Solution:
    def snakesAndLadders(self, board: List[List[int]]) -> int:
        
        board.reverse()                                     
        for i in range(1,len(board),2): board[i].reverse()
        arr = [None]+list(chain(*board))                    
                                                            
                                                            
        n, queue, seen, ct = len(arr)-1, deque([1]), {1}, 0               

        while queue:                                        
            lenQ = len(queue)

            for _ in range(lenQ):                           

                cur = queue.popleft()
                if cur == n: return ct

                for i in range(cur+1, min(cur+7,n+1)):      
                    nxt = arr[i] if arr[i]+1 else i         

                    if nxt in seen: continue                
                    seen.add(nxt)
                    queue.append(nxt)                       
                    
            ct += 1                    
        
        return -1


class Solution:
    def minMutation(self, startGene: str, endGene: str, bank: List[str]) -> int:
        n = len(startGene)
        def difference(gene1, gene2):
            one_diff = False
            for i in range(n):
                if gene1[i] != gene2[i]:
                    if one_diff:
                        return False
                    one_diff = True
            return True
            
        seen = {startGene}
        queue = deque([startGene])
        mutations = 0
        while queue:
            mutations += 1
            for i in range(len(queue)):
                gene = queue.popleft()
                for mutation in bank:
                    if mutation not in seen and difference(gene,mutation):
                        if mutation == endGene:
                            return mutations
                        queue.append(mutation)
                        seen.add(mutation)
        return -1


class Solution:
    def minMutation(self, startGene: str, endGene: str, bank: List[str]) -> int:
        
        
        bankSet = set(bank)
        
        
        options = ['A', 'C', 'G', 'T']
        
        
        queue = deque()
        queue.append(startGene)
        
        
        visited = set()
        visited.add(startGene)
        
        
        count = 0
        
        
        while queue:
            size = len(queue)
            for i in range(size):
                gene = queue.popleft()
                if gene == endGene:
                    return count
                for j in range(8):
                    for option in options:
                        newGene = gene[:j] + option + gene[j+1:]
                        if newGene in bankSet and newGene not in visited:
                            visited.add(newGene)
                            queue.append(newGene)
            count += 1
        
        
        return -1

class Solution:
    def minMutation(self, start: str, end: str, bank: List[str]) -> int:
        dic=defaultdict(lambda :0)
        lst=[[start,0]]
        dic[start]=1
        while lst:
            x,d=lst.pop(0)
            if x==end:
                return d
            for i in range(len(bank)):
                ct=0
                for j in range(8):
                    if x[j]!=bank[i][j]:
                        ct+=1
                if ct==1:
                    if dic[bank[i]]==0:
                        lst.append([bank[i],d+1])
                        dic[bank[i]]=1
        return -1

class Solution:
    def minMutation(self, start: str, end: str, bank: list[str]) -> int:
        bank = set(bank) | {start}

        def dfs(st0, cnt):
            if st0 == end:
                return cnt

            bank.remove(st0)
            for i, ch0 in enumerate(st0):
                for ch1 in "ACGT":
                    if (
                        ch0 != ch1
                        and (st1 := st0[:i] + ch1 + st0[i + 1 :]) in bank
                        and (res := dfs(st1, cnt + 1)) != -1
                    ):
                        return res

            return -1

        return dfs(start, 0)
    

class Solution:
    def minMutation(self,start, end, bank):
        bank_set = set(bank)  
        if end not in bank_set:
            return -1
        
        queue = deque([(start, 0)])  
        visited = set([start])  

        while queue:  
            current_gene, mutations = queue.popleft()  

            if current_gene == end:  
                return mutations

            for i in range(len(current_gene)):  
                for c in ['A', 'C', 'G', 'T']:  
                    next_gene = current_gene[:i] + c + current_gene[i+1:]

                    if next_gene in bank_set and next_gene not in visited:  
                        visited.add(next_gene)
                        queue.append((next_gene, mutations + 1))

        return -1
    

def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
    def difference(word1, word2):
        one_diff = False
        for i in range(len(word1)):
            if word1[i] != word2[i]:
                if one_diff:
                    return False
                one_diff = True
        return True
        
    seen = {beginWord}
    queue = deque([beginWord])
    sequence = 1
    while queue:
        sequence += 1
        for i in range(len(queue)):
            word = queue.popleft()
            for change in wordList:
                if change not in seen and difference(word,change):
                    if change == endWord:
                        return sequence
                    queue.append(change)
                    seen.add(change)
    return 0

class Solution:
    def ladderLength(self, start: str, end: str, wordList: List[str]) -> int:
        bank_set = set(wordList)  
        if end not in bank_set:
            return 0
        
        queue = deque([(start, 1)]) 
        visited = set([start]) 
        alphabet_set = set(string.ascii_lowercase)
        while queue:  
            current_word, changes = queue.popleft()  

            if current_word == end: 
                return changes

            for i in range(len(current_word)):  
                for c in alphabet_set:  
                    next_word = current_word[:i] + c + current_word[i+1:]

                    if next_word in bank_set and next_word not in visited:  
                        visited.add(next_word)
                        queue.append((next_word, changes + 1))

        return 0
    

class Solution:
    def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:

        if endWord not in wordList or not endWord or not beginWord or not wordList:
            return 0
        L = len(beginWord)
        all_combo_dict = defaultdict(list)
        for word in wordList:
            for i in range(L):
                all_combo_dict[word[:i] + "*" + word[i+1:]].append(word) 
        queue = deque([(beginWord, 1)])
        visited = set()
        visited.add(beginWord)
        while queue:
            current_word, level = queue.popleft()
            for i in range(L):
                intermediate_word = current_word[:i] + "*" + current_word[i+1:]
                for word in all_combo_dict[intermediate_word]:
                    if word == endWord:
                        return level + 1
                    if word not in visited:
                        visited.add(word)
                        queue.append((word, level + 1))
        return 0


def hashword(word):
    h=0
    c=1
    for i in word:
        h+=c*(ord(i)-97)
        c=c*26
    return h
class Solution:
    def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
        d={}
        for i in range(len(wordList)):
            d[hashword(wordList[i])]=0
        h=hashword(beginWord)
        q=deque()
        q.append(h)
        f=hashword(endWord)
        q=deque()
        q.append([h,1])
        if h in d:
            d[h]=1
        while q:
            x=q.popleft()
            h=x[0]
            ans=x[1]
            if h==f:
                return ans
            c=1
            for i in range(10):
                for j in range(26):
                    y=h-(((h%(c*26))//c)*c)+(c*j)
                    if y in d:
                        if d[y]==0:
                            d[y]=1
                            q.append([y,ans+1])
                c=c*26
        return 0


class Trie:

    def __init__(self):
        self.children = {}
        

    def insert(self, word: str) -> None:
        node = self.children
        for char in word:
            if char not in node:
                node[char] = {}
            node = node[char] 
        node["END"] = "END"      

    def search(self, word: str) -> bool:
        node = self.children
        for char in word:
            if char not in node:
                return False
            node = node[char]
        return "END" in node
        

    def startsWith(self, prefix: str) -> bool:
        node = self.children
        for char in prefix:
            if char not in node:
                return False
            node = node[char]
        return True
        
class TrieNode:
    def __init__(self):
        
        self.children = {}
        self.isEnd = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        cur = self.root
        
        for c in word:
            
            if c not in cur.children:
                cur.children[c] = TrieNode()
            cur = cur.children[c]
        cur.isEnd = True
        

    def search(self, word: str) -> bool:
        cur = self.root
        
        for c in word:
            
            if c not in cur.children:
                return False
            cur = cur.children[c]
        return cur.isEnd
        

    def startsWith(self, prefix: str) -> bool:
        
        cur = self.root
        for c in prefix:
            if c not in cur.children:
                return False
            cur = cur.children[c]
        return True



class TrieNode:
    def __init__(self):
        
        self.children = {}
        self.isEnd = False

class WordDictionary:

    def __init__(self):
        self.root = TrieNode()
        

    def addWord(self, word: str) -> None:
        cur = self.root
        
        for c in word:
            
            if c not in cur.children:
                cur.children[c] = TrieNode()
            cur = cur.children[c]
        cur.isEnd = True 
        

    def search(self, word: str) -> bool:
        nodes = [self.root]

        for c in word:
            if not nodes:
                return False
            all_nodes = []
            for node in nodes:
                if c in node.children:
                    all_nodes.append(node.children[c])
                elif c == ".":
                    all_nodes.extend(list(node.children.values()))
            nodes = all_nodes
        for node in nodes:
            if node.isEnd:
                return True
        return False

class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_word = False
        
class WordDictionary:
    def __init__(self):
        self.root = TrieNode()      

    def addWord(self, word):
        current_node = self.root
        for character in word:
            current_node = current_node.children.setdefault(character, TrieNode())
        current_node.is_word = True
        
    def search(self, word):
        def dfs(node, index):
            if index == len(word):
                return node.is_word
               
            if word[index] == ".":
                for child in node.children.values():
                    if dfs(child, index+1):
                        return True
                    
            if word[index] in node.children:
                return dfs(node.children[word[index]], index+1)
            
            return False
    
        return dfs(self.root, 0)
    

class WordDictionary:
        def __init__(self):

            self.words = defaultdict(list)


        def addWord(self, word: str) -> None:

            self.words[len(word)].append(word)


        def search(self, word: str) -> bool:

            n = len(word)

            if '.' in word:
                
                for w in self.words[n]:
                    if all(word[i] in (w[i], '.') for i in range(n)):
                        return True

                else: return False

            return word in self.words[n]

class WordDictionary:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.trie = {}

    def addWord(self, word: str) -> None:
        """
        Adds a word into the data structure.
        """
        root = self.trie
        for char in word:
            if char not in root:
                root[char] = {}
            root = root[char]
        root['isEnd'] = True

    def search(self, word: str) -> bool:
        """
        Returns if the word is in the data structure. A word could contain the dot character '.' to represent any one letter.
        """
        return self.dfs(word, 0, self.trie)

    def dfs(self, word: str, index: int, node: dict) -> bool:
        if index == len(word):
            return node.get('isEnd', False)
        
        if word[index] == '.':
            for key in node:
                if key != 'isEnd' and self.dfs(word, index + 1, node[key]):
                    return True
        else:
            if word[index] in node:
                return self.dfs(word, index + 1, node[word[index]])
        
        return False


class WordDictionary:

    def __init__(self):
        self.trie = {}

    def addWord(self, word: str) -> None:
        root = self.trie
        for char in word:
            if char not in root:
                root[char] = {}
            root = root[char]
        root['isEnd'] = True


class Solution:
    def findWords(self, board: List[List[str]], words: List[str]) -> List[str]:
        word_dict = WordDictionary()
        for word in words:
            word_dict.addWord(word)
        words = set()
        for i in range(len(board)):
            for j in range(len(board[0])):
                self.dfs(board,i,j,word_dict.trie,'',words,set())
        return list(words)


    def dfs(self, board,row,col, node,word, found,visited):
        if board[row][col] not in node:
            return 
        visited.add((row,col))
        word += board[row][col]
        if node[board[row][col]].get('isEnd', False):
            found.add(word)
        for i in [row-1, row+1]:
            if i < len(board) and i > -1 and (i,col) not in visited:
                self.dfs(board,i,col,node[board[row][col]],word, found,visited)
        for i in [col-1, col+1]:
            if i < len(board[0]) and i > -1 and (row,i) not in visited:
                self.dfs(board,row,i,node[board[row][col]],word,found,visited)
        visited.remove((row,col))

class WordDictionary:

    def __init__(self):
        self.trie = {}

    def addWord(self, word: str) -> None:
        root = self.trie
        for char in word:
            if char not in root:
                root[char] = {}
            root = root[char]
        root['isEnd'] = word


class Solution:
    def findWords(self, board: List[List[str]], words: List[str]) -> List[str]:
        word_dict = WordDictionary()
        for word in words:
            word_dict.addWord(word)
        words = set()
        
        def dfs(row,col, node):
            if board[row][col] not in node:
                return 
            letter = board[row][col]
            board[row][col] = ""
            if 'isEnd' in node[letter]:
                words.add(node[letter]['isEnd'])
            for i in [row-1, row+1]:
                if i < len(board) and i > -1:
                    dfs(i,col,node[letter])
            for i in [col-1, col+1]:
                if i < len(board[0]) and i > -1:
                    dfs(row,i,node[letter])
            board[row][col] = letter

        for i in range(len(board)):
            for j in range(len(board[0])):
                dfs(i,j,word_dict.trie)
        return words


    

from functools import reduce
from collections import defaultdict
class Solution:
    def findWords(self, board: List[List[str]], words: List[str]) -> List[str]:
        
        
        Trie = lambda: defaultdict(Trie)
        trie = Trie()
        END = True
        
        for word in words:
            reduce(dict.__getitem__,word,trie)[END] = word
        
        res = set()
        def findstr(i,j,t):
            if END in t:
                res.add(t[END])
            letter = board[i][j]
            board[i][j] = ""
            if i > 0 and board[i-1][j] in t:
                findstr(i-1,j,t[board[i-1][j]])
            if j>0 and board[i][j-1] in t:
                findstr(i,j-1,t[board[i][j-1]])
            if i < len(board)-1 and board[i+1][j] in t:
                findstr(i+1,j,t[board[i+1][j]])
            if j < len(board[0])-1 and board[i][j+1] in t:
                findstr(i,j+1,t[board[i][j+1]])
            board[i][j] = letter
            
            return 
        
        for i, row in enumerate(board):
            for j, char in enumerate(row):
                if board[i][j] in trie:
                    findstr(i,j,trie[board[i][j]])
        return res

class WordDictionary:

    def __init__(self):
        self.trie = {}

    def addWord(self, word: str) -> None:
        root = self.trie
        for char in word:
            if char not in root:
                root[char] = {}
            root = root[char]
        root['isEnd'] = word


class Solution:
    def findWords(self, board: List[List[str]], words: List[str]) -> List[str]:
        word_dict = WordDictionary()
        for word in words:
            word_dict.addWord(word)
        words = set()
        
        def dfs(row,col, node):
            if board[row][col] not in node:
                return 
            letter = board[row][col]
            board[row][col] = ""
            if 'isEnd' in node[letter]:
                words.add(node[letter]['isEnd'])

            if row > 0:
                dfs(row-1,col,node[letter])
            if col > 0:
                dfs(row,col-1,node[letter])
            if row < len(board)-1:
                dfs(row+1,col,node[letter])
            if col < len(board[0])-1:
                dfs(row,col+1,node[letter])
            
            board[row][col] = letter

        for i in range(len(board)):
            for j in range(len(board[0])):
                dfs(i,j,word_dict.trie)
        return words

class WordDictionary:

    def __init__(self):
        self.trie = {}

    def addWord(self, word: str) -> None:
        root = self.trie
        for char in word:
            if char not in root:
                root[char] = {}
            root = root[char]
        root['isEnd'] = word


class Solution:
    def findWords(self, board: List[List[str]], words: List[str]) -> List[str]:
        word_dict = WordDictionary()
        for word in words:
            word_dict.addWord(word)
        words = set()
        
        def dfs(row,col, node):
            letter = board[row][col]
            board[row][col] = ""
            if 'isEnd' in node:
                words.add(node['isEnd'])

            if row > 0 and board[row-1][col] in node:
                dfs(row-1,col,node[board[row-1][col]])
            if col > 0 and board[row][col-1] in node:
                dfs(row,col-1,node[board[row][col-1]])
            if row < len(board)-1 and board[row+1][col] in node:
                dfs(row+1,col,node[board[row+1][col]])
            if col < len(board[0])-1 and board[row][col+1] in node:
                dfs(row,col+1,node[board[row][col+1]])
            
            board[row][col] = letter

        for i in range(len(board)):
            for j in range(len(board[0])):
                if board[i][j] in word_dict.trie:
                    dfs(i,j,word_dict.trie[board[i][j]])
        return words

class Solution:
    def findWords(self, board: List[List[str]], words: List[str]) -> List[str]:
        
        
        def dfs(x, y, root):
            
            letter = board[x][y]
            
            cur = root[letter]
            
            word = cur.pop('#', False)
            if word:
                
                res.append(word)
            
            board[x][y] = '*'
            
            for dirx, diry in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                curx, cury = x + dirx, y + diry
                
                if 0 <= curx < m and 0 <= cury < n and board[curx][cury] in cur:
                    dfs(curx, cury, cur)
            
            board[x][y] = letter
            
            if not cur:
                root.pop(letter)
                
        
        trie = {}
        for word in words:
            cur = trie
            for letter in word:
                cur = cur.setdefault(letter, {})
            cur['#'] = word
            
        
        m, n = len(board), len(board[0])
        
        res = []
        
        
        for i in range(m):
            for j in range(n):
                
                if board[i][j] in trie:
                    dfs(i, j, trie)
        
        
        return res

class WordDictionary:

    def __init__(self):
        self.trie = {}

    def addWord(self, word: str) -> None:
        root = self.trie
        for char in word:
            if char not in root:
                root[char] = {}
            root = root[char]
        root['isEnd'] = word


class Solution:
    def findWords(self, board: List[List[str]], words: List[str]) -> List[str]:
        word_dict = WordDictionary()
        for word in words:
            word_dict.addWord(word)

        words = []
        
        def dfs(row,col, node):
            
            letter = board[row][col]
            cur = node[letter]
            board[row][col] = ""
            if 'isEnd' in cur:
                words.append(cur.pop('isEnd'))

            if row > 0 and board[row-1][col] in cur:
                dfs(row-1,col,cur)
            if col > 0 and board[row][col-1] in cur:
                dfs(row,col-1,cur)
            if row < len(board)-1 and board[row+1][col] in cur:
                dfs(row+1,col,cur)
            if col < len(board[0])-1 and board[row][col+1] in cur:
                dfs(row,col+1,cur)
            
            board[row][col] = letter
            if not cur:
                node.pop(letter)

        for i in range(len(board)):
            for j in range(len(board[0])):
                if board[i][j] in word_dict.trie:
                    dfs(i,j,word_dict.trie)
        return words







class Trie:
    def __init__(self, words: List[str] = None):
		
        self.root = {"length": 0}
        for word in words:
            self.insert(word)
			
	
    def __len__(self) -> int:
        return self.root["length"]

    
    def insert(self, word: str) -> None:
        current = self.root
        for c in word:
            if c not in current:
                current[c] = {"length": 0}
            
            current["length"] += 1
            current = current[c]
        current["length"] += 1
        current["?"] = True

    
    def remove(self, word: str) -> None:
        current = self.root
        current["length"] -= 1
        for i, c in enumerate(word):
            if c in current:
                current[c]["length"] -= 1
                if current[c]["length"] < 1:
                    current.pop(c)
                    break
                else:
                    current = current[c]
        
        if i == len(word) - 1 and "?" in current:
            current.pop("?")

    
    
	
    def contains(self, word: List[str]) -> int:
        current = self.root
        for c in word:
            if c not in current:
                return 0
            current = current[c]
        return 2 if "?" in current else 1





















class Solution:
    def findWords(self, board: List[List[str]], words: List[str]) -> List[str]:
        NUM_ROWS, NUM_COLS = len(board), len(board[0])
        
        
        seq_two = set()
        candidates = []
        reversed_words = set()
        
        
        for i in range(NUM_ROWS):
            for j in range(NUM_COLS - 1):
                seq_two.add(board[i][j] + board[i][j + 1])
        for j in range(NUM_COLS):
            for i in range(NUM_ROWS - 1):
                seq_two.add(board[i][j] + board[i + 1][j])
        
        for word in words:
            in_board = True
            for i in range(len(word) - 1):
                
                
                if (
                    word[i : i + 2] not in seq_two
                    and word[i + 1] + word[i] not in seq_two
                ):
                    in_board = False
                    break
            if not in_board:
                continue
            
            
            if word[:4] == word[0] * 4:
                word = word[::-1]
                reversed_words.add(word)
            candidates.append(word)

        NUM_ROWS, NUM_COLS = len(board), len(board[0])
        
        res = set()
        
        
        
        trie = Trie(candidates)
        
        
        def dfs(row: int, col: int, current: List[str]) -> None:
            current.append(board[row][col])
            board[row][col] = "."
            found = trie.contains(current)
            
            
            if not found:
                board[row][col] = current.pop()
                return
            
            if found == 2:
                w = "".join(current)
                if w in reversed_words:
                    res.add(w[::-1])
                    reversed_words.remove(w)
                else:
                    res.add(w)
                trie.remove(w)
            
            dirs = ((0, 1), (0, -1), (1, 0), (-1, 0))
            for di, dj in dirs:
                i, j = row + di, col + dj
                if (
                    0 <= i < NUM_ROWS
                    and 0 <= j < NUM_COLS
                    and board[i][j] != "."
                ):
                    dfs(i, j, current)
            
            board[row][col] = current.pop()

        for i in range(NUM_ROWS):
            for j in range(NUM_COLS):
                dfs(i, j, [])
        return res


class Solution:
    def letterCombinations(self, digits: str) -> List[str]:
        letters = ["abc","def","ghi","jkl","mno","pqrs","tuv","wxyz"]
        output = []
        for digit in digits:
            temp = output
            output = []
            for letter in letters[int(digit)-2]:
                if not temp:
                    output += [letter]
                else: 
                    for combination in temp:
                        output += [combination + letter]
        return output

class Solution:
    def letterCombinations(self, digits: str) -> List[str]:
        if not digits:
            return []
        
        phone = {"2": "abc", "3": "def", "4": "ghi", "5": "jkl", "6": "mno", "7": "pqrs", "8": "tuv", "9": "wxyz"}
        res = []
        
        def backtrack(combination, next_digits):
            if not next_digits:
                res.append(combination)
                return
            
            for letter in phone[next_digits[0]]:
                backtrack(combination + letter, next_digits[1:])
        
        backtrack("", digits)
        return res
    
class Solution:
    def letterCombinations(self, digits: str) -> List[str]:
        result = []
        if not digits:
            return result
        
        mapping = {
            '2': 'abc',
            '3': 'def',
            '4': 'ghi',
            '5': 'jkl',
            '6': 'mno',
            '7': 'pqrs',
            '8': 'tuv',
            '9': 'wxyz'
        }
        
        def backtrack(index, current_combination):
            nonlocal result
            if index == len(digits):
                result.append(current_combination)
                return
            
            digit = digits[index]
            letters = mapping[digit]
            for letter in letters:
                backtrack(index + 1, current_combination + letter)
        
        backtrack(0, "")
        return result


class Solution:
    def combine(self, n: int, k: int) -> List[List[int]]:
        output = []
        
        def dfs(start, array):
            if len(array) == k:
                output.append(array)
                return 
            for i in range (start+1,n+1):
                dfs(i, array + [i] )
        dfs(0,[])
        return output

class Solution:  
    def combine(self, n, k):   
        sol=[]
        def backtrack(remain,comb,nex):
            
            if remain==0:
                sol.append(comb.copy())
            else:
                
                for i in range(nex,n+1):
                    
                    comb.append(i)
                    
                    backtrack(remain-1,comb,i+1)
                    
                    comb.pop()
            
        backtrack(k,[],1)
        return sol

class Solution:
    def combine(self, n: int, k: int) -> List[List[int]]:
        def backtrack(first = 1, curr = []):
            if len(curr) == k:
                output.append(curr[:])
                return
            for i in range(first, n + 1):
                curr.append(i)
                backtrack(i + 1, curr)
                curr.pop()
        output = []
        backtrack()
        return output

class Solution:
    def combine(self, n: int, k: int) -> List[List[int]]:
        def generate_combinations(elems, num):
            elems_tuple = tuple(elems)
            total = len(elems_tuple)
            if num > total:
                return
            curr_indices = list(range(num))
            while True:
                yield tuple(elems_tuple[i] for i in curr_indices)
                for idx in reversed(range(num)):
                    if curr_indices[idx] != idx + total - num:
                        break
                else:
                    return
                curr_indices[idx] += 1
                for j in range(idx+1, num):
                    curr_indices[j] = curr_indices[j-1] + 1

        return [list(combination) for combination in generate_combinations(range(1, n+1), k)]


class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        full = len(nums)
        visited = set()
        def backtrack(curr = []):
            if len(curr) == full:
                output.append(curr[:])
                return
            for num in nums:
                if num not in visited:
                    visited.add(num)
                    curr.append(num)
                    backtrack(curr)
                    curr.pop()
                    visited.remove(num)
        output = []
        backtrack()
        return output

class Solution:
    def permute(self, l: List[int]) -> List[List[int]]:
        def dfs(path, used, res):
            if len(path) == len(l):
                res.append(path[:]) 
                return

            for i, letter in enumerate(l):
                
                if used[i]:
                    continue
                
                path.append(letter)
                used[i] = True
                dfs(path, used, res)
                
                path.pop()
                used[i] = False
            
        res = []
        dfs([], [False] * len(l), res)
        return res


class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        added = set()
        def backtrack(curr = [],total = 0):
            if total == target:
                current = tuple(sorted(curr))
                if current not in added:
                    added.add(current)
                    output.append(curr[:])
                return
            elif total > target:
                return
            for num in candidates:
                curr.append(num)
                backtrack(curr, total + num)
                curr.pop()
        output = []
        backtrack()
        return output
        
class Solution:
    def combinationSum(self, candidates, target):
        res = []
        candidates.sort()
        self.dfs(candidates, target, 0, [], res)
        return res
        
    def dfs(self, nums, target, index, path, res):
        if target < 0:
            return  
        if target == 0:
            res.append(path)
            return 
        for i in range(index, len(nums)):
            self.dfs(nums, target-nums[i], i, path+[nums[i]], res)


class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        dp = [[] for _ in range(target+1)]
        for c in candidates:                                  
            for i in range(c, target+1):                      
                if i == c: dp[i].append([c])
                for comb in dp[i-c]: dp[i].append(comb + [c]) 
        return dp[-1]

class Solution:
    def combinationSum(self, candidates, target):
        res = []
        candidates.sort()
        self.dfs(candidates, target, 0, [], res)
        return res
        
    def dfs(self, nums, target, index, path, res):
        if target < 0:
            return True  
        if target == 0:
            res.append(path)
            return 
        for i in range(index, len(nums)):
            if self.dfs(nums, target-nums[i], i, path+[nums[i]], res):
                break  

class Solution:
    def totalNQueens(self, n: int) -> int:
        locations = set()
        rows = set(range(n))
        used = set()
        self.count = 0

        def dfs(col,level):
            if len(used) == n:
                self.count += 1
            for row in rows:
                if row not in used and safe(row,col):
                    locations.add((row,col))
                    used.add(row)
                    dfs(col+1,level+1)
                    used.remove(row)
                    locations.remove((row,col))

        def safe(row, col):
            temp_row = row
            temp_col = col
            while row >= 0 and col >= 0:
                row -= 1
                col -= 1
                if (row,col) in locations:
                    return False
            row = temp_row
            col = temp_col
            while row < n and col < n:
                row += 1
                col -= 1
                if (row,col) in locations:
                    return False
            return True
        dfs(0,0)
        return self.count


class Solution:
    def totalNQueens(self, n: int) -> int:
        locations = set()
        rows = set(range(n))
        used = set()
        self.count = 0

        def dfs(col,level):
            if len(used) == n:
                self.count += 1
            for row in rows:
                if row not in used and safe(row,col):
                    locations.add((row,col))
                    used.add(row)
                    dfs(col+1,level+1)
                    used.remove(row)
                    locations.remove((row,col))

        def safe(row, col):
            for r, c in locations:
                if abs(row - r) == abs(col - c):
                    return False
            return True
        dfs(0,0)
        return self.count


class Solution:
    def totalNQueens(self, n: int) -> int:
        state=[['.'] * n for _ in range(n)]
		
		
        visited_cols=set()
		
		
        
        
        
        
        
        
        visited_diagonals=set()
		
		 
        
        
        
        
        
        
        
        visited_antidiagonals=set()
        
        res=set()
        def backtrack(r):
            if r==n:
                res.add(map('#'.join, map(''.join, state))) 
                return
                        
            for c in range(n):
			 
                if not(c in visited_cols or (r-c) in visited_diagonals or (r+c) in visited_antidiagonals):
                    visited_cols.add(c)
                    visited_diagonals.add(r-c)
                    visited_antidiagonals.add(r+c)
                    state[r][c]='Q'
                    backtrack(r+1)
                    
					
                    visited_cols.remove(c)
                    visited_diagonals.remove(r-c)
                    visited_antidiagonals.remove(r+c)
                    state[r][c]='.'
                        
        backtrack(0)
        return len(res)

class Solution:
    def totalNQueens(self, N: int) -> int:
        self.ans = 0
        
        def place(i: int, vert: int, ldiag: int, rdiag:int) -> None:
            if i == N: self.ans += 1
            else:
                for j in range(N):
                    vmask, lmask, rmask = 1 << j, 1 << (i+j), 1 << (N-i-1+j)
                    if vert & vmask or ldiag & lmask or rdiag & rmask: continue
                    place(i+1, vert | vmask, ldiag | lmask, rdiag | rmask)
            
        place(0,0,0,0)
        return self.ans
        

class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        m = 2*n
        output = []
        self.val = 0
        self.length = 0
        curr = []
        def dfs():
            if self.length == m:
                output.append(''.join(curr))
                return
            if self.val > 0:
                curr.append(')')
                self.val -= 1
                self.length +=1
                dfs()
                curr.pop()
                self.val +=1
                self.length -=1
            if self.val < m-self.length:
                curr.append('(')
                self.val +=1 
                self.length +=1
                dfs()
                curr.pop()
                self.val -=1
                self.length -=1

        dfs()
        return output

class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        row_length = len(board)-1
        col_length = len(board[0])-1
        word_length = len(word) - 1
        self.word_found = False
        def dfs(row,col,index):
            if board[row][col] != word[index]:
                return
            if index == word_length:
                self.word_found = True
                return
            temp = board[row][col]
            board[row][col] = ""
            if row > 0:
                dfs(row-1,col,index+1)
            if col > 0 and not self.word_found:
                dfs(row,col-1,index+1)
            if row < row_length and not self.word_found:
                dfs(row+1,col,index+1)
            if col < col_length and not self.word_found:
                dfs(row,col+1,index+1)
            board[row][col] = temp
        for i in range(row_length+1):
            for j in range(col_length+1):
                if not self.word_found:
                    dfs(i,j,0)
        return self.word_found

class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        
        R = len(board)
        C = len(board[0])
        
        if len(word) > R*C:
            return False
        
        count = Counter(sum(board, []))
        
        for c, countWord in Counter(word).items():
            if count[c] < countWord:
                return False
            
        if count[word[0]] > count[word[-1]]:
             word = word[::-1]
                        
        seen = set()
        
        def dfs(r, c, i):
            if i == len(word):
                return True
            if r < 0 or c < 0 or r >= R or c >= C or word[i] != board[r][c] or (r,c) in seen:
                return False
            
            seen.add((r,c))
            res = (
                dfs(r+1,c,i+1) or 
                dfs(r-1,c,i+1) or
                dfs(r,c+1,i+1) or
                dfs(r,c-1,i+1) 
            )
            seen.remove((r,c))  

            return res
        
        for i in range(R):
            for j in range(C):
                if dfs(i,j,0):
                    return True
        return False
    

class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        row_length = len(board)-1
        col_length = len(board[0])-1
        word_length = len(word) - 1
        self.word_found = False

        if len(word) > (row_length+1)*(col_length+1):
            return False
        
        count = Counter(sum(board, []))
        
        for c, countWord in Counter(word).items():
            if count[c] < countWord:
                return False
            
        if count[word[0]] > count[word[-1]]:
             word = word[::-1]

        def dfs(row,col,index):
            if board[row][col] != word[index]:
                return
            if index == word_length:
                self.word_found = True
                return
            temp = board[row][col]
            board[row][col] = ""
            if row > 0:
                dfs(row-1,col,index+1)
            if col > 0 and not self.word_found:
                dfs(row,col-1,index+1)
            if row < row_length and not self.word_found:
                dfs(row+1,col,index+1)
            if col < col_length and not self.word_found:
                dfs(row,col+1,index+1)
            board[row][col] = temp
        for i in range(row_length+1):
            for j in range(col_length+1):
                if not self.word_found:
                    dfs(i,j,0)
        return self.word_found
        





class Solution:
    def sortedArrayToBST(self, nums: List[int]) -> Optional[TreeNode]:
        if not nums:
            return None
        if (length := len(nums)) == 1:
            return TreeNode(nums[0])
        parent = TreeNode(nums[length // 2])
        parent.left = self.sortedArrayToBST(nums[:length // 2])
        parent.right = self.sortedArrayToBST(nums[(length // 2)+1:])
        return parent
        
def sortedArrayToBST(self, nums: List[int]) -> Optional[TreeNode]:
    def rec(nums, start, end):
        if start <= end:
            mid = (start + end) // 2
            node = TreeNode(nums[mid])
            node.left = rec(nums, start, mid - 1)
            node.right = rec(nums, mid + 1, end)
            return node
    return rec(nums, 0, len(nums) - 1)


class Solution:
    def sortList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if not head:
            return None
        length = 0
        dummy = ListNode()
        dummy.next = head
        while head:
            length +=1 
            head = head.next
        
        def divide(start,length):
            if length == 1:
                start.next = None
                return start
            mid = length//2
            prev, curr = None, start
            for _ in range(mid):
                prev, curr = curr, curr.next
            prev.next = None
            
            return merge(divide(start,mid), divide(curr, length - mid))

        
        def merge(list1,list2):
            new_dummy = ListNode()
            current = new_dummy
            while list1 and list2:
                if list1.val <= list2.val:
                    current.next = list1
                    list1 = list1.next
                else:
                    current.next = list2
                    list2 = list2.next
                current = current.next
            current.next = list1 or list2
            return new_dummy.next
        
 

        return divide(dummy.next,length)
        
        
        
class Solution:
    def sortList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if not head or not head.next:
            return head
        
        
        left = head
        right = self.getMid(head)
        tmp = right.next
        right.next = None
        right = tmp
        
        left = self.sortList(left)
        right = self.sortList(right)
        
        return self.merge(left, right)
    
    def getMid(self, head):
        slow = head
        fast = head.next
        
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        return slow
    
    
    def merge(self, list1, list2):
        newHead = tail = ListNode()
        while list1 and list2:
            if list1.val > list2.val:
                tail.next = list2
                list2 = list2.next
            else:
                tail.next = list1
                list1 = list1.next
            tail = tail.next
        
        if list1:
            tail.next = list1
        if list2:
            tail.next = list2
        
        return newHead.next
        


class Solution:
    def sortList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        dummy = ListNode(0)
        dummy.next = head

        
        steps = 1
        while True:
            
            
            prev = dummy

            
            remaining = prev.next

            
            num_loops = 0
            while remaining:
                num_loops += 1

                
                sublists = [None, None]
                sublists_tail = [None, None]
                for i in range(2):
                    sublists[i] = remaining
                    substeps = steps
                    while substeps and remaining:
                        substeps -= 1
                        sublists_tail[i] = remaining
                        remaining = remaining.next
                    
                    if sublists_tail[i]:
                        sublists_tail[i].next = None

                
                
                while sublists[0] and sublists[1]:
                    if sublists[0].val <= sublists[1].val:
                        prev.next = sublists[0]
                        sublists[0] = sublists[0].next
                    else:
                        prev.next = sublists[1]
                        sublists[1] = sublists[1].next
                    prev = prev.next

                
                if sublists[0]:
                    prev.next = sublists[0]
                    prev = sublists_tail[0]
                else:
                    prev.next = sublists[1]
                    prev = sublists_tail[1]

            
            steps *= 2

            
            if 1 >= num_loops:
                return dummy.next




class Solution:
    def sortList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        buffer_size = 8
        dummy = ListNode(0)
        dummy.next = head

        
        steps = 1
        while True:
            prev = dummy
            remaining = prev.next

            
            num_loops = 0
            while remaining:
                num_loops += 1

                
                sublists = [None] * buffer_size
                sublists_tail = [None] * buffer_size
                for i in range(buffer_size):
                    sublists[i] = remaining
                    substeps = steps
                    while substeps and remaining:
                        substeps -= 1
                        sublists_tail[i] = remaining
                        remaining = remaining.next
                    if sublists_tail[i]:
                        sublists_tail[i].next = None

                
                num_sublists = buffer_size
                while 1 < num_sublists:
                    subdummy = ListNode()
                    for i in range(0, num_sublists, 2):
                        subprev = subdummy
                        subprev.next = None
                        while sublists[i] and sublists[i + 1]:
                            if sublists[i].val <= sublists[i + 1].val:
                                subprev.next = sublists[i]
                                sublists[i] = sublists[i].next
                            else:
                                subprev.next = sublists[i + 1]
                                sublists[i + 1] = sublists[i + 1].next
                            subprev = subprev.next

                        if sublists[i]:
                            subprev.next = sublists[i]
                            sublists_tail[i // 2] = sublists_tail[i]
                        else:
                            subprev.next = sublists[i + 1]
                            sublists_tail[i // 2] = sublists_tail[i + 1]

                        sublists[i // 2] = subdummy.next

                    num_sublists //= 2

                prev.next = sublists[0]
                prev = sublists_tail[0]

            steps *= buffer_size

            if 1 >= num_loops:
                return dummy.next


class Solution(object):
    def sortList(self, head):
        if head is None:
            return None
        ptr=head
        arr=[]
        while ptr is not None:
            arr.append(ptr.val)
            ptr=ptr.next
        arr.sort()
        n = ListNode(arr[0])
        head=n
        temp=head
        for i in range(1,len(arr)):
            n1 = ListNode(arr[i])
            temp.next=n1
            temp=temp.next       
        return head


class Solution(object):
    def sortList(self, head):
        if head is None or head.next is None:
            return head

        length = self.getLength(head)
        dummy = ListNode(0)
        dummy.next = head

        step = 1
        while step < length:
            curr = dummy.next
            tail = dummy

            while curr:
                left = curr
                right = self.split(left, step)
                curr = self.split(right, step)

                tail = self.merge(left, right, tail)

            step *= 2

        return dummy.next

    def getLength(self, head):
        length = 0
        curr = head
        while curr:
            length += 1
            curr = curr.next
        return length

    def split(self, head, step):
        if head is None:
            return None

        for i in range(1, step):
            if head.next is None:
                break
            head = head.next

        right = head.next
        head.next = None
        return right

    def merge(self, left, right, tail):
        curr = tail
        while left and right:
            if left.val < right.val:
                curr.next = left
                left = left.next
            else:
                curr.next = right
                right = right.next
            curr = curr.next

        curr.next = left if left else right
        while curr.next:
            curr = curr.next

        return curr


class Solution:
    def construct(self, grid: List[List[int]]) -> 'Node':
        if len(grid) == 1:
            return Node(grid[0][0],True)
        length = len(grid) // 2
        Nodes = []
        matrix1 = [row[:length] for row in grid[:length]]
        matrix2= [row[length:] for row in grid[:length]]
        matrix3 = [row[:length] for row in grid[length:]]
        matrix4 = [row[length:] for row in grid[length:]]
        grids = [matrix1,matrix2,matrix3,matrix4]
        for grid in grids:
            Nodes.append(self.construct(grid))
        if Nodes[0].val != -1 and all(node.val == Nodes[0].val for node in Nodes):
            return Nodes[0]
        return Node(-1,False,Nodes[0],Nodes[1],Nodes[2],Nodes[3])



class Node:
    def __init__(self, val, isLeaf, topLeft, topRight, bottomLeft, bottomRight):
        self.val = val
        self.isLeaf = isLeaf
        self.topLeft = topLeft
        self.topRight = topRight
        self.bottomLeft = bottomLeft
        self.bottomRight = bottomRight


class Solution:
    def construct(self, grid: List[List[int]]) -> 'Node':     
        
        def helper(row,col,length):
            if length == 1:
                return Node(grid[row][col],True)
            new_length = length // 2
            TL = helper(row,col,new_length)
            TR= helper(row,col + new_length,new_length)
            BL = helper(row+new_length,col,new_length)
            BR = helper(row+new_length,col+new_length,new_length)

            if TL.isLeaf and TL.val == TR.val == BL.val == BR.val:
                return TL
            return Node(-1,False,TL,TR,BL,BR)


        return helper(0,0,len(grid))



class Solution:
    def construct(self, grid: List[List[int]]) -> 'Node':
        
        def allSame(i, j, w):
            for x in range(i, i + w):
                for y in range(j, j + w):
                    if grid[x][y] != grid[i][j]:
                        return False
            return True

        def helper(i, j, w):
            if allSame(i, j, w):
                return Node(grid[i][j], True)
            new_length = w//2
            node = Node(True, False)
            node.topLeft = helper(i, j, new_length)
            node.topRight = helper( i, j + new_length, new_length)
            node.bottomLeft = helper( i + new_length, j, new_length)
            node.bottomRight = helper( i + new_length, j + new_length, new_length)
            return node

        return helper(0, 0, len(grid))



class Solution:
    def construct(self, grid: List[List[int]]) -> Node:
        return self.helper(grid, 0, 0, len(grid))

    def helper(self, grid, i, j, w):
        if self.allSame(grid, i, j, w):
            return Node(grid[i][j] == 1, True)

        node = Node(True, False)
        node.topLeft = self.helper(grid, i, j, w // 2)
        node.topRight = self.helper(grid, i, j + w // 2, w // 2)
        node.bottomLeft = self.helper(grid, i + w // 2, j, w // 2)
        node.bottomRight = self.helper(grid, i + w // 2, j + w // 2, w // 2)
        return node

    def allSame(self, grid, i, j, w):
        for x in range(i, i + w):
            for y in range(j, j + w):
                if grid[x][y] != grid[i][j]:
                    return False
        return True


class Solution:
    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        def merge2lists(head1,head2):
            dummy = ListNode()
            output = dummy
            while head1 and head2:
                if head1.val <= head2.val:
                    output.next = head1
                    head1 = head1.next
                elif head2.val < head1.val:
                    output.next = head2
                    head2 = head2.next
                output = output.next
            output.next = head1 or head2
            return dummy.next

        length = len(lists)
        while length >1:
            for i in range(0,length,2):
                if i+1 == length:
                    lists[i//2] = merge2lists(lists[i],None)
                    continue
                lists[i//2] = merge2lists(lists[i],lists[i+1])
            length = math.ceil(length/2)
        return lists[0] if length != 0 else None


class Solution:
    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        ListNode.__eq__ = lambda self, other: self.val == other.val
        ListNode.__lt__ = lambda self, other: self.val < other.val
        h = []
        head = tail = ListNode(0)
        for i in lists:
            if i:
                heapq.heappush(h, i)

        while h:
            node = heapq.heappop(h)
            tail.next = node
            tail = tail.next
            if node.next:
                heapq.heappush(h,  node.next)

        return head.next
        
class Solution:
    def merge(self, left: ListNode, right: ListNode) -> ListNode:
        dummy = ListNode(-1)
        temp = dummy
        while left and right:
            if left.val < right.val:
                temp.next = left
                temp = temp.next
                left = left.next
            else:
                temp.next = right
                temp = temp.next
                right = right.next
        while left:
            temp.next = left
            temp = temp.next
            left = left.next
        while right:
            temp.next = right
            temp = temp.next
            right = right.next
        return dummy.next
    
    def mergeSort(self, lists: List[ListNode], start: int, end: int) -> ListNode:
        if start == end:
            return lists[start]
        mid = start + (end - start) // 2
        left = self.mergeSort(lists, start, mid)
        right = self.mergeSort(lists, mid + 1, end)
        return self.merge(left, right)
    
    def mergeKLists(self, lists: List[ListNode]) -> ListNode:
        if not lists:
            return None
        return self.mergeSort(lists, 0, len(lists) - 1)



class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        Output = nums[0]
        current = nums[0]
        for i in range (1,len(nums)):
            current = max(current + nums[i], nums[i])
            if current > Output:
                Output = current
        return Output


def maxSubArray(self, nums: List[int]) -> int:
    maxSum = float('-inf')
    currentSum = 0
    
    for num in nums:
        currentSum += num
        
        if currentSum > maxSum:
            maxSum = currentSum
        
        if currentSum < 0:
            currentSum = 0
    
    return maxSum

def maxSubArray(self, nums: List[int]) -> int:
    maxSum = nums[0]
    currentSum = nums[0]

    for num in nums[1:]:
        currentSum = max(num, currentSum + num)
        maxSum = max(maxSum, currentSum)

    return maxSum



def maxSubarraySumCircular(self, A: List[int]) -> int:
    if max(A) <= 0:
        return max(A)
    
    max_sum = curr_max = min_sum = curr_min = A[0] 
    
    for i in range(1, len(A)): 
        curr_max = max(A[i], curr_max + A[i]) 
        max_sum = max(max_sum, curr_max)
        curr_min = min(A[i], curr_min + A[i]) 
        min_sum = min(min_sum, curr_min)
        
    return max(max_sum, sum(A) - min_sum)


def maxSubarraySumCircular(self, nums: List[int]) -> int:
    total_sum = 0
    curr = 0
    max_sum = float('-inf')
    flag = 1
    ans = float('-inf')
    
    for i in nums:
        if i >= 0:
            flag = 0
            break
        ans = max(ans, i)
    
    if flag:
        return ans
    
    for i in nums:
        total_sum += i
        curr += i
        max_sum = max(max_sum, curr)
        if curr < 0:
            curr = 0
    min_sum = float('inf')
    curr = 0
    
    for i in nums:
        curr += i
        min_sum = min(min_sum, curr)
        if curr > 0:
            curr = 0
    
    ans2 = total_sum - min_sum
    return max(max_sum, ans2)


class Solution:
    def searchInsert(self, nums: List[int], target: int) -> int:
        
        def divide(start,end):
            if end == start:
                if nums[start] > target:
                    return start
                if nums[start] < target:
                    return start + 1
                return start
            mid = (start+end)//2

            if  nums[mid] > target:
                return divide(start,mid)
            elif nums[mid] < target:
                return divide(mid+1,end)
            else:
                return mid
        return divide(0,len(nums)-1)

def searchInsert(self, nums: List[int], target: int) -> int:
    low, high = 0, len(nums)
    while low < high:
        mid = (low + high) // 2
        if target > nums[mid]:
            low = mid + 1
        else:
            high = mid
    return low

import bisect

class Solution:
    def searchInsert(self, nums: List[int], target: int) -> int:
        return bisect.bisect_left(nums, target)


class Solution:
    def searchInsert(self, nums: List[int], target: int) -> int:
        if not nums:
            return 0
        
        for i, num in enumerate(nums):
            if num >= target:
                return i
        
        return len(nums)


class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        
        low, high = 0, len(matrix) - 1
        
        while low < high:
            mid = (low + high) // 2
            if target == matrix[mid][0]:
                return True
            elif target > matrix[mid][0]:
                if target < matrix[mid+1][0]:
                    low = mid
                    high = mid
                else:    
                    low = mid + 1
            else:
                high = mid - 1
        
        if matrix[low][0] == target:
            return True

        row = low
        low, high = 0, len(matrix[0]) - 1
        while low <= high:
            mid = (low + high) // 2
            if target == matrix[row][mid]:
                return True
            elif target > matrix[row][mid]:
                low = mid + 1
            else:
                high = mid - 1
        return False

        
class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        if not matrix:
            return False
        m, n = len(matrix), len(matrix[0])
        left, right = 0, m * n - 1

        while left <= right:
            mid = (left + right) // 2
            mid_row, mid_col = divmod(mid, n)

            if matrix[mid_row][mid_col] == target:
                return True
            elif matrix[mid_row][mid_col] < target:
                left = mid + 1
            else:
                right = mid - 1

        return False
        
class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        return any(target in row for row in matrix)
    
class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        for row in matrix:
            if row[-1] >= target:
                return target in row
        return False

class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        r = bisect.bisect_left(matrix, target, key=lambda row: row[-1])  
        return r < len(matrix) and matrix[r][bisect.bisect_left(matrix[r], target)] == target

class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        n = len(matrix[0])
        def get(idx: int) -> int:
            r, c = divmod(idx, n)
            return matrix[r][c]
        return get(bisect.bisect_left(range(len(matrix)*n-1), target, key=get)) == target


class Solution:
    def findPeakElement(self, nums: List[int]) -> int:
        low, high = 0, len(nums) - 1
        while low <= high:
            mid = (low + high) // 2
            middle = nums[(low + high) // 2]
            if mid + 1 <= len(nums)-1:
                compare = nums[mid+1]
            else:
                compare = -math.inf
            if middle < compare:
                low = mid + 1
                continue
            if mid - 1 >= 0:
                compare = nums[mid-1]
            else:
                compare = -math.inf
            if middle < compare:
                high = mid - 1
            else:
                return mid

class Solution:
    def findPeakElement(self, nums: List[int]) -> int:
        low, high = 0, len(nums) - 1
        while low <= high:
            mid = (low + high) // 2
            middle = nums[mid]
            compare_r = nums[mid + 1] if mid + 1 <= len(nums) - 1 else -math.inf
            compare_l = nums[mid - 1] if mid - 1 >= 0 else -math.inf
            if middle < compare_r:
                low = mid + 1
            elif middle < compare_l:
                high = mid - 1
            else:
                return mid
            
class Solution:
    def findPeakElement(self, nums: List[int]) -> int:
        left =0
        right = len(nums)-1
        while left < right:
            mid = left + (right - left ) //2
            if nums[mid] > nums[mid+1]: 
                right = mid 
            else:
                left = mid +1
        return left

class Solution:
    def findPeakElement(self, nums: List[int]) -> int:
        left, right = 0, len(nums)-1
        
        while left < right:
            mid = (left + right) // 2
            if nums[mid] > nums[mid+1]:
                right = mid
            else:
                left = mid + 1
                
        return left

class Solution:
    def search(self, nums: List[int], target: int) -> int:
        left, right, end = 0, len(nums)-1 , nums[-1]
        
        while left < right:
            mid = (left + right) // 2
            if nums[mid] < end:
                right = mid  
            else:
                left = mid + 1
                
        pivot = left

        left, right, length = 0, len(nums)-1 , len(nums)
        
        while left <= right:
            mid = ((left + right) // 2)
            real_mid = (mid + pivot)%length
            if target == nums[real_mid]:
                return real_mid
            elif target > nums[real_mid]:
                left = mid + 1
            else:
                right = mid - 1
        return -1



class Solution:
    def search(self, nums: List[int], target: int) -> int:
        left, right = 0, len(nums) - 1

        while left <= right:
            mid = (left + right) // 2

            if nums[mid] == target:
                return mid

            
            if nums[left] <= nums[mid]:
                if nums[left] <= target < nums[mid]:
                    right = mid - 1
                else:
                    left = mid + 1
            
            else:
                if nums[mid] < target <= nums[right]:
                    left = mid + 1
                else:
                    right = mid - 1

        return -1


class Solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        low, high = 0, len(nums) - 1
        left = right = -1
        while low <= high:
            mid = (low + high) // 2
            if target == nums[mid]:
                compare_r = nums[mid + 1] if mid + 1 <= len(nums) - 1 else -math.inf
                if nums[mid] == compare_r:
                    low = mid + 1
                else: 
                    right = mid
                    break
            elif target > nums[mid]:
                low = mid + 1
            else:
                high = mid - 1
        
        low, high = 0, len(nums) - 1
        while low <= high:
            mid = (low + high) // 2
            if target == nums[mid]:
                compare_l = nums[mid - 1] if mid - 1 >= 0 else -math.inf
                if nums[mid] == compare_l:
                    high = mid-1
                else:
                    left = mid
                    break
            elif target > nums[mid]:
                low = mid + 1
            else:
                high = mid - 1
        
        return [left,right]
        
class Solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        
        def search(x):
            lo, hi = 0, len(nums)           
            while lo < hi:
                mid = (lo + hi) // 2
                if nums[mid] < x:
                    lo = mid+1
                else:
                    hi = mid                    
            return lo
        
        lo = search(target)
        hi = search(target+1)-1
        
        if lo <= hi:
            return [lo, hi]
                
        return [-1, -1]
        

class Solution:
    def findMin(self, nums: List[int]) -> int:
        left, right, end = 0, len(nums)-1 , nums[-1]
        
        while left < right:
            mid = (left + right) // 2
            if nums[mid] < end:
                right = mid  
            else:
                left = mid + 1
        return nums[left]
        
class Solution:
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        if len(nums1) > len(nums2):
            nums1, nums2 = nums2, nums1
        low, high = 0, len(nums1)
        len1, len2 = len(nums1), len(nums2)
        Total = (len1 + len2)
        while low <= high:
            split1 = (low + high) // 2
            split2 =  (Total+1)//2 - split1

            maxX = -math.inf if split1 == 0 else nums1[split1 - 1]
            maxY = -math.inf if split2 == 0 else nums2[split2 - 1]

            minX = math.inf if split1 == len1 else nums1[split1]
            minY = math.inf if split2 == len2 else nums2[split2]

            if maxX <= minY and maxY <= minX:
                return (max(maxX, maxY) + min(minX, minY)) / 2 if (len1 + len2) % 2 == 0 else max(maxX, maxY)
            elif minX < maxY:
                low = split1 + 1
            else:
                high = split1 - 1

class Solution:
    def findMedianSortedArrays(self, nums1, nums2):
        
        merged = nums1 + nums2

        
        merged.sort()

        
        total = len(merged)

        if total % 2 == 1:
            
            return float(merged[total // 2])
        else:
            
            middle1 = merged[total // 2 - 1]
            middle2 = merged[total // 2]
            return (float(middle1) + float(middle2)) / 2.0
        
class Solution:
    def findMedianSortedArrays(self, nums1, nums2):
        n = len(nums1)
        m = len(nums2)
        i = 0
        j = 0
        m1 = 0
        m2 = 0

        
        for count in range(0, (n + m) // 2 + 1):
            m2 = m1
            if i < n and j < m:
                if nums1[i] > nums2[j]:
                    m1 = nums2[j]
                    j += 1
                else:
                    m1 = nums1[i]
                    i += 1
            elif i < n:
                m1 = nums1[i]
                i += 1
            else:
                m1 = nums2[j]
                j += 1

        
        if (n + m) % 2 == 1:
            return float(m1)
        else:
            ans = float(m1) + float(m2)
            return ans / 2.0
        

class Solution:
    def findMedianSortedArrays(self, nums1, nums2):
        n1 = len(nums1)
        n2 = len(nums2)
        
        
        if n1 > n2:
            return self.findMedianSortedArrays(nums2, nums1)
        
        n = n1 + n2
        left = (n1 + n2 + 1) // 2 
        low = 0
        high = n1
        
        while low <= high:
            mid1 = (low + high) // 2 
            mid2 = left - mid1 
            
            l1 = float('-inf')
            l2 = float('-inf')
            r1 = float('inf')
            r2 = float('inf')
            
            
            if mid1 < n1:
                r1 = nums1[mid1]
            if mid2 < n2:
                r2 = nums2[mid2]
            if mid1 - 1 >= 0:
                l1 = nums1[mid1 - 1]
            if mid2 - 1 >= 0:
                l2 = nums2[mid2 - 1]
            
            if l1 <= r2 and l2 <= r1:
                
                if n % 2 == 1:
                    return max(l1, l2)
                else:
                    return (max(l1, l2) + min(r1, r2)) / 2.0
            elif l1 > r2:
                
                high = mid1 - 1
            else:
                
                low = mid1 + 1
        
        return 0 


class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        heap = []
        for num in nums:
            heapq.heappush(heap, num)
            if len(heap) > k:
                heapq.heappop(heap)
        return heapq.heappop(heap)
    
def findKthLargest(self, nums: List[int], k: int) -> int:
    heap = []
    for num in nums:
        heapq.heappush(heap, -num)
    output = None
    for i in range(k):
        output = -heapq.heappop(heap)
    return output

class Solution:
    def findKthLargest(self, nums, k):
        return sorted(nums, reverse=True)[k-1]

class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        heap = nums[:k]
        heapq.heapify(heap)
        
        for num in nums[k:]:
            if num > heap[0]:
                heapq.heappop(heap)
                heapq.heappush(heap, num)
        
        return heap[0]


class Solution:
    def findKthLargest(self, nums, k):
        left, right = 0, len(nums) - 1
        while True:
            pivot_index = random.randint(left, right)
            new_pivot_index = self.partition(nums, left, right, pivot_index)
            if new_pivot_index == len(nums) - k:
                return nums[new_pivot_index]
            elif new_pivot_index > len(nums) - k:
                right = new_pivot_index - 1
            else:
                left = new_pivot_index + 1

    def partition(self, nums, left, right, pivot_index):
        pivot = nums[pivot_index]
        nums[pivot_index], nums[right] = nums[right], nums[pivot_index]
        stored_index = left
        for i in range(left, right):
            if nums[i] < pivot:
                nums[i], nums[stored_index] = nums[stored_index], nums[i]
                stored_index += 1
        nums[right], nums[stored_index] = nums[stored_index], nums[right]
        return stored_index


class MinHeap:
    def __init__(self):
        self.heap = []
    
    def append(self,val):
        self.heap.append(val)
        self.bubbleUp()
    
    def pop(self):
        top = self.heap[0]
        end = self.heap.pop()
        if len(self.heap) > 0 :
            self.heap[0] = end
            self.bubbleDown()
        
        return top
    
    def peek(self):
        return self.heap[0]
    
    def bubbleUp(self):
        idx = len(self.heap) - 1
        element = self.heap[idx]
        while idx > 0 :
            parentIdx = (idx - 1) // 2
            parent = self.heap[parentIdx]
            if element >= parent:
                break
            self.heap[parentIdx] = element
            self.heap[idx] = parent
            idx = parentIdx
        
    
    def bubbleDown(self):
        idx = 0
        length = len(self.heap)
        element = self.heap[0]
        while True:
            leftChildIdx = 2 * idx + 1
            rightChildIdx = 2 * idx + 2
            swap = None
            if leftChildIdx < length:
                leftChild = self.heap[leftChildIdx]
                if leftChild < element :
                    swap = leftChildIdx
            if rightChildIdx < length:
                rightChild = self.heap[rightChildIdx]
                if (not swap and rightChild < element) or (swap and rightChild < leftChild) :
                    swap = rightChildIdx
            if not swap:
                break
            self.heap[idx] = self.heap[swap]
            self.heap[swap] = element
            idx = swap

class Solution:
    def findKthLargest(self, nums, k):
        heap = MinHeap()
        
        
        for i in range(k):
            heap.append(nums[i])
            
        
        for i in range(k, len(nums)):
            if nums[i] > heap.peek():
                heap.pop()
                heap.append(nums[i])
        
        return heap.peek()



class Solution:
    def findMaximizedCapital(self, k: int, w: int, profits: List[int], capital: List[int]) -> int:
        
        combined = list(zip(capital, profits))
        heap_profit = []
        heapq.heapify(combined)
        while k:
            k -= 1
            while combined and combined[0][0] <= w:
                heapq.heappush(heap_profit,-heapq.heappop(combined)[1])
            if heap_profit:
                w += -heapq.heappop(heap_profit)
            else:
                break
        return w
    
class Solution:
    def findMaximizedCapital(self, k: int, w: int, profits: List[int], capital: List[int]) -> int:
        n = len(profits)
        projects = [(capital[i], profits[i]) for i in range(n)]
        projects.sort()
        i = 0
        maximizeCapital = []
        while k > 0:
            while i < n and projects[i][0] <= w:
                heapq.heappush(maximizeCapital, -projects[i][1])
                i += 1
            if not maximizeCapital:
                break
            w -= heapq.heappop(maximizeCapital)
            k -= 1
        return w


class Solution:
    def kSmallestPairs(self, nums1: List[int], nums2: List[int], k: int) -> List[List[int]]:
        heap = [(nums1[0]+nums2[0],0,0)]
        length1 = len(nums1)
        length2 = len(nums2)
        output = []
        while heap and len(output) < k:
            total, i, j = heapq.heappop(heap)
            output.append([nums1[i],nums2[j]])
            if j + 1 < length2:
                heapq.heappush(heap,(nums1[i]+nums2[j+1],i,j+1))
            if j == 0 and i+1 < length1:
                heapq.heappush(heap,(nums1[i+1]+nums2[0],i+1,0))
        return output

class Solution:
    def kSmallestPairs(self, nums1, nums2, k):
        resV = []  
        pq = []  

        
        for x in nums1:
            heapq.heappush(pq, [x + nums2[0], 0])  

        
        while k > 0 and pq:
            pair = heapq.heappop(pq)
            s, pos = pair[0], pair[1]  

            resV.append([s - nums2[pos], nums2[pos]])  

            
            if pos + 1 < len(nums2):
                heapq.heappush(pq, [s - nums2[pos] + nums2[pos + 1], pos + 1])

            k -= 1  

        return resV  


    def kSmallestPairs(self, nums1: List[int], nums2: List[int], k: int) -> List[List[int]]:
        res = []
        from heapq import heappush, heappop
        m, n, visited = len(nums1), len(nums2), set()
        if m == 0 or n == 0: return [] 
        h = [(nums1[0]+nums2[0], (0, 0))]
        for _ in range(min(k, (m*n))):
            val, (i, j) = heappop(h)
            res.append([nums1[i], nums2[j]])
            if i+1 < m and (i+1, j) not in visited:
                heappush(h, (nums1[i+1]+nums2[j], (i+1, j)))
                visited.add((i+1, j))
            if j+1 < n and (i, j+1) not in visited:
                heappush(h, (nums1[i]+nums2[j+1], (i, j+1)))
                visited.add((i, j+1))
        return res


class MedianFinder:

    def __init__(self):
        self.lower = []
        self.upper = []

    def addNum(self, num: int) -> None:
        if not self.lower:
            heapq.heappush(self.lower,-num)
        elif len(self.lower) != len(self.upper):
            if -self.lower[0] <= num:
                heapq.heappush(self.upper,num)
            else:
                heapq.heappush(self.upper,-heapq.heappop(self.lower))
                heapq.heappush(self.lower,-num)
        else:
            if self.upper[0] > num:
                heapq.heappush(self.lower,-num)
            else:
                heapq.heappush(self.lower,-heapq.heappop(self.upper))
                heapq.heappush(self.upper,num)
        
    def findMedian(self) -> float:
        if len(self.lower) == len(self.upper):
            return (-self.lower[0] + self.upper[0])/2
        else:
            return -self.lower[0]



class MedianFinder:
    def __init__(self):
        self.lower = []  
        self.upper = []  

    def addNum(self, num):
        
        heapq.heappush(self.lower, -num)
        
        
        heapq.heappush(self.upper, -heapq.heappop(self.lower))
        
        
        if len(self.lower) < len(self.upper):
            heapq.heappush(self.lower, -heapq.heappop(self.upper))
            
    def findMedian(self):
        if len(self.lower) > len(self.upper):
            return -self.lower[0]                  
        else:
            return (self.upper[0] - self.lower[0]) / 2  
        

class Solution:
    def addBinary(self, a: str, b: str) -> str:
        i, j = len(a) - 1, len(b) - 1
        output = ''
        carry = 0
        while i >= 0 or j >=0:
            total = carry
            if i>=0:
                total += int(a[i])
            if j>= 0:
                total += int(b[j])
            carry = total // 2
            output = str(total%2) + output
            i -= 1
            j -= 1
        if carry:
            return "1" + output
        return output
 

class Solution:
  def addBinary(self, a: str, b: str) -> str:
    s = []
    carry = 0
    i = len(a) - 1
    j = len(b) - 1

    while i >= 0 or j >= 0 or carry:
      if i >= 0:
        carry += int(a[i])
        i -= 1
      if j >= 0:
        carry += int(b[j])
        j -= 1
      s.append(str(carry % 2))
      carry //= 2

    return ''.join(reversed(s))

class Solution:
    def addBinary(self, a: str, b: str) -> str:
        aL, bL = -len(a), -len(b)
        i, carry, res = -1, 0, ""

        while i >= aL or i >= bL:
            aBit = int(a[i]) if i >= aL else 0
            bBit = int(b[i]) if i >= bL else 0
            
            sum = aBit + bBit + carry
            res = str(sum % 2) + res
            carry = sum // 2

            i -= 1
            
        return "1" + res if carry else res

class Solution:   
    def addBinary(self,a: str, b: str) -> str:
        
        sum_int = int(a, 2) + int(b, 2)
        
        
        return bin(sum_int)[2:]


class Solution:
    def reverseBits(self, n: int) -> int:
        n = bin(n)[2:].zfill(32)
        return int(''.join(list(str(n))[::-1]),2)
    

class Solution:
    def reverseBits(self, n: int) -> int:
        return int(bin(n)[2:].zfill(32)[::-1],2)
        

class Solution:
    def reverseBits(self, n: int) -> int:
        res = 0
        for _ in range(32):
            res = (res<<1) + (n&1)
            n>>=1
        return res

class Solution:
    def reverseBits(self, n: int) -> int:
        res = 0
        for _ in range(32):
            res = (res<<1) + (n&1)
            n>>=1
        return res

class Solution:
    def hammingWeight(self, n: int) -> int:
        res = 0
        while n != 0:
            res += (n&1)
            n>>=1
        return res

class Solution:
    def hammingWeight(self, n: int) -> int:
        return Counter(bin(n)[2:])['1']

def hammingWeight(self, n):
        count = 0
        while n:
            if n & 1: count += 1
            n = n >> 1
        return count

class Solution:
    def hammingWeight(self, n):
        return bin(n)[2:].count('1')

class Solution:
    def hammingWeight(self,n):
        count = 0
        while n:
            n &= (n - 1)
            count += 1
        return count

class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        res = 0
        for num in nums:
            res ^= num
        return res

def singleNumber(self, nums: List[int]) -> int:
	return reduce(lambda total, el: total ^ el, nums)


class Solution:
    def singleNumber(self,nums):
        ones = 0
        twos = 0

        for num in nums:
            twos = twos | (ones & num)
            ones = ones ^ num
            common_bit_mask = ~(ones & twos)
            ones = ones & common_bit_mask
            twos = twos & common_bit_mask

        return ones

class Solution:
    def singleNumber(self, nums):
        ones = 0
        twos = 0
        for num in nums:
            ones = (ones ^ num) & ~twos
            twos = (twos ^ num) & ~ones
        return ones

class Solution:
    def singleNumber(self, nums):
        count = defaultdict(int)
        
        for x in nums:
            count[x] += 1

        for x, freq in count.items():
            if freq == 1:
                return x
        
        return -1


class Solution:
    def singleNumber(self, nums):
        ans = 0

        for i in range(32):
            bit_sum = 0
            for num in nums:
                
                if num < 0:
                    num = num & (2**32-1)
                bit_sum += (num >> i) & 1
            bit_sum %= 3
            ans |= bit_sum << i

        
        if ans >= 2**31:
            ans -= 2**32

        return ans


class Solution:
    def rangeBitwiseAnd(self, left: int, right: int) -> int:
        shift = 0
        while left < right:
            left >>= 1
            right >>= 1
            shift += 1
        return left << shift

class Solution:
    def rangeBitwiseAnd(self, left: int, right: int) -> int:
        if (left > 0 and right > 0) and math.log2(right) - math.log2(left) >= 1:
            return 0
        shift = 0
        while left < right:
            left >>= 1
            right >>= 1
            shift += 1
        return left << shift



class Solution:
    def isPalindrome(self, x: int) -> bool:
        if x < 0:
            return False
        if int(str(x)[::-1]) == x:
            return True
        return False

class Solution:
    def isPalindrome(self, x: int) -> bool:
        if x < 0:
            return False
        y = 0
        check = x
        while x:
            y = x%10 + 10*y
            x //= 10

        return check==y

def isPalindrome(self, x: int) -> bool:
    if x < 0:
        return False
    
    return str(x) == str(x)[::-1]

class Solution:
    def isPalindrome(self, x: int) -> bool:
        if x < 0 or (x > 0 and x%10 == 0):   
            return False
        
        result = 0
        while x > result:
            result = result * 10 + x % 10
            x = x // 10
            
        return (x == result or x == result // 10) 


class Solution:
    def plusOne(self, digits: List[int]) -> List[int]:
        for i in range(len(digits)-1,-1,-1):
            if digits[i] != 9:
                digits[i] += 1
                break
            digits[i] = 0
        if digits[i] == 0:
            return [1] + digits
        return digits
        
class Solution:
    def plusOne(self, digits: List[int]) -> List[int]:
        return [int(d) for d in str(int(''.join(map(str, digits))) + 1)]
        


class Solution:
    def plusOne(self, digits: List[int]) -> List[int]:
        if digits[-1] < 9:
            digits[-1] += 1
            return digits
        elif len(digits) == 1 and digits[0] == 9:
            return [1, 0]
        else:
            digits[-1] = 0
            digits[0:-1] = self.plusOne(digits[0:-1])
            return digits

class Solution:
    def plusOne(self, digits):
        strings = ""
        for number in digits:
            strings += str(number)

        temp = str(int(strings) +1)

        return [int(temp[i]) for i in range(len(temp))]

class Solution:
    def trailingZeroes(self, n: int) -> int:
        count = 0
        while n > 0:
            n = n // 5
            count += n
        return count

class Solution:
    def trailingZeroes(self, n):
        x   = 5
        res = 0
        while x <= n:
            res += n//x
            x   *= 5
        return res

class Solution:
    def trailingZeroes(self, n: int) -> int:
        fact = math.factorial(n) 

        string = [i for i in str(fact)] 
        
        string.reverse() 
        count = 0
        
        
        for number in string:
            if number != '0':
                return count
            count += 1

class Solution:
    def trailingZeroes(self, n: int) -> int:
        if n < 5: 
            return 0
        else: 
            return int(n/5) + self.trailingZeroes(int(n/5))



class Solution:
    def mySqrt(self, x: int) -> int:
        if not x:
            return 0
        for i in range (0,x):
            if i*i > x:
                return i-1
        return 1
        
class Solution:
    def mySqrt(self, x: int) -> int:
        low = 0
        high = x
        while low < high:
            mid = (low+high)//2
            squared = mid*mid
            if squared == x:
                return mid
            elif squared > x:
                high = mid - 1
            else:
                low = mid + 1
        return low if low*low <= x else low -1

class Solution:
    def mySqrt(self, x: int) -> int:
        if x == 0:
            return 0
        first, last = 1, x
        while first <= last:
            mid = first + (last - first) // 2
            if mid == x // mid:
                return mid
            elif mid > x // mid:
                last = mid - 1
            else:
                first = mid + 1
        return last
    

class Solution:
    def mySqrt(self, x: int) -> int:
        res = 1
        for i in range(20):
            temp = res
            res = temp - (temp**2 - x)/(2 * temp)
        return math.floor(res)

class Solution:
    def myPow(self, x: float, n: int) -> float:
        
        def power_helper(x, n):
            result = 1.0
            current_product = x
            
            while n > 0:
                if n % 2 == 1:
                    result *= current_product
                current_product *= current_product
                n //= 2
            
            return result
        
        if n == 0:
            return 1.0
        elif n > 0:
            return power_helper(x, n)
        else:
            return 1 / power_helper(x, -n)

class Solution:
    def myPow(self, x: float, n: int) -> float:

        def function(base=x, exponent=abs(n)):
            if exponent == 0:
                return 1
            elif exponent % 2 == 0:
                return function(base * base, exponent // 2)
            else:
                return base * function(base * base, (exponent - 1) // 2)

        f = function()
        
        return float(f) if n >= 0 else 1/f

class Solution:
    def myPow(self,a, b):
        if not a:
            return 0
        flag = 0

        
        if a < 0:
            a = abs(a)
            if b % 2 != 0:
                flag = 1

            res = math.exp(b * math.log(a))
            return res * -1 if flag == 1 else res

        
        else:
            return math.exp(b * math.log(a))


class Solution:
    def maxPoints(self, points: List[List[int]]) -> int:
        output = 0
        for xy1 in points:
            slopes = defaultdict(int)
            for xy2 in points:
                if xy1 == xy2:
                    continue
                if xy2[0] -xy1[0] != 0:    
                    slope = (xy2[1]-xy1[1]) / (xy2[0] - xy1[0])
                else:
                    slope = math.inf
                slopes[slope] += 1
                output = max(slopes[slope],output)
        return output + 1



        
class Solution:
    def maxPoints(self, points: List[List[int]]) -> int:
        output = 1
        for xy1 in points:
            slopes = defaultdict(lambda : 1)
            for xy2 in points:
                if xy1 == xy2:
                    continue
                if xy2[0] -xy1[0] != 0:    
                    slope = (xy2[1]-xy1[1]) / (xy2[0] - xy1[0])
                else:
                    slope = math.inf
                slopes[slope] += 1
                output = max(slopes[slope],output)
        return output
        

class Solution:
    def maxPoints(self, points: List[List[int]]) -> int:
        if len(points) <= 2:
            return len(points)
        
        def find_slope(p1, p2):
            x1, y1 = p1
            x2, y2 = p2
            if x1-x2 == 0:
                return math.inf
            return (y1-y2)/(x1-x2)
        
        ans = 1
        for i, p1 in enumerate(points):
            slopes = defaultdict(int)
            for j, p2 in enumerate(points[i+1:]):
                slope = find_slope(p1, p2)
                slopes[slope] += 1
                ans = max(slopes[slope], ans)
        return ans+1
    


class Solution:
    def maxPoints(self, points: list[list[int]]) -> int:
                                                

        points.sort()                           
        slope, M = defaultdict(int), 0          
                                                
        for i, (x1, y1) in enumerate(points):   
                                                
            slope.clear()                       
                                                
            for x2, y2 in points[i + 1:]:       
                dx, dy = x2 - x1, y2 - y1
                                                
                G = math.gcd(dx, dy)                 
                m = (dx//G,dy//G)
                
                slope[m] += 1
                if slope[m] > M: M = slope[m]
    
        return M + 1


class Solution:
    def climbStairs(self, n: int) -> int:
        if not n:
            return 1
        if n == -1:
            return 0
        score = 0
        for i in range(1,3):
            score += self.climbStairs(n-i)
        return score
    
class Solution:
    def climbStairs(self, n: int) -> int:
        if n == 0 or n == 1:
            return 1
        return self.climbStairs(n-1) + self.climbStairs(n-2)

class Solution:
    def __init__(self):
        self.memo = {}  
    
    def climbStairs(self, n: int) -> int:
        if n == 0:
            return 1
        if n == -1:
            return 0

        
        if n in self.memo:
            return self.memo[n]

        score = 0
        for i in range(1, 3):  
            score += self.climbStairs(n - i)
        
        
        self.memo[n] = score

        return score

        
    
class Solution:
    def climbStairs(self, n: int) -> int:
        memo = {}
        return self.helper(n, memo)
    
    def helper(self, n: int, memo: dict[int, int]) -> int:
        if n == 0 or n == 1:
            return 1
        if n not in memo:
            memo[n] = self.helper(n-1, memo) + self.helper(n-2, memo)
        return memo[n]

class Solution:
    def climbStairs(self, n: int) -> int:
        if n == 0 or n == 1:
            return 1

        dp = [0] * (n+1)
        dp[0] = dp[1] = 1
        
        for i in range(2, n+1):
            dp[i] = dp[i-1] + dp[i-2]
        return dp[n]

class Solution:
    def climbStairs(self, n: int) -> int:
        if n == 0 or n == 1:
            return 1
        prev, curr = 1, 1
        for i in range(2, n+1):
            temp = curr
            curr = prev + curr
            prev = temp
        return curr

def climbStairs(self, n):
    a = b = 1
    for _ in range(n):
        a, b = b, a + b
    return a

class Solution:
    def __init__(self):
        self.matrix = [[0, 1], [1, 1]]
        self.identity = [[1, 0], [0, 1]]

    def mul(self, matrix1, matrix2):
        a1, a2 = matrix1[0]
        a3, a4 = matrix1[1]
        b1, b2 = matrix2[0]
        b3, b4 = matrix2[1]
        return [
            [a1 * b1 + a2 * b3, a1 * b2 + a2 * b4],
            [a3 * b1 + a4 * b3, a3 * b2 + a4 * b4]
        ]

    def climbStairs(self, n):
        result = self.identity
        bits = bin(n + 1)[2:]  

        for bit in bits:
            result = self.mul(result, result)
            if bit == "1":
                result = self.mul(result, self.matrix)

        return result[1][0]


class Solution:
    def rob(self, nums: List[int]) -> int:
        if len(nums) == 1:
            return nums[0]
        DP = [0]*len(nums)
        DP[0] = nums[0]
        DP[1] = max(nums[0],nums[1])
        for i in range(2,len(nums)):
            DP[i] = max(DP[i-1],DP[i-2]+nums[i])
        return DP[-1]

class Solution:
    def rob(self, nums: List[int]) -> int:
        prev = 0
        curr = 0
        for num in nums:
            curr, prev = max(curr,prev+num), curr
        return curr

from itertools import islice

class Solution:
    def rob(self, nums: List[int]) -> int:
        if not nums:
            return 0
        
        if len(nums) == 1:
            return nums[0]
        
        nums.append(0)
        nums.reverse()
        for idx, num in enumerate(islice(nums, 3, None), 3):
            nums[idx] = max(num + nums[idx - 2], num + nums[idx - 3])
            
        return max(nums[-1], nums[-2])

class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        wordDict = set(wordDict)
        DP = [False]*(len(s)+1)
        DP[0] = True
        for i in range(1,len(s)+1):
            for j in range(0,i):
                if DP[j] and s[j:i] in wordDict:
                    DP[i] = True
                    break
        return DP[-1]
                    
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        wordDict = set(wordDict)
        word_locations=[0]
        for i in range(1,len(s)+1):
            for index in word_locations:
                if s[index:i] in wordDict:
                    word_locations.append(i)
                    break
        return True if word_locations[-1] == len(s) else False
                    


class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        
        def construct(current,wordDict, memo={}):
            if current in memo:
                return memo[current]

            if not current:
                return True

            for word in wordDict:
                if current.startswith(word):
                    new_current = current[len(word):]
                    if construct(new_current,wordDict,memo):
                        memo[current] = True
                        return True

            memo[current] = False
            return False

        return construct(s,wordDict)


class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        n = len(s)
        dp = [False] * (n + 1)
        dp[0] = True
        max_len = max(map(len, wordDict))  

        for i in range(1, n + 1):
            for j in range(i - 1, max(i - max_len - 1, -1), -1): 
                if dp[j] and s[j:i] in wordDict:
                    dp[i] = True
                    break

        return dp[n]



class Solution:
    def __init__(self):
        self.seen = {}
    def coinChange(self, coins: List[int], amount: int) -> int:
        coins.sort(reverse=True)
        def helper(amount):
            if amount < 0:
                return -1
            elif amount in self.seen:
                return self.seen[amount]
            elif amount == 0:
                return 0
            minimum = math.inf
            for coin in coins:
                outcome = helper(amount-coin)
                if outcome >= 0:
                    minimum = min(minimum,outcome)
            if minimum == math.inf:
                self.seen[amount] = -1
            else:
                self.seen[amount] = minimum + 1
            return  self.seen[amount]
        return helper(amount)

        
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:        
        dp=[math.inf] * (amount+1)
        dp[0]=0
        
        for coin in coins:
            for i in range(coin, amount+1):
                if i-coin>=0:
                    dp[i]=min(dp[i], dp[i-coin]+1)
        
        return -1 if dp[-1]==math.inf else dp[-1]


class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        numCoins = len(coins)
        
        
        minCoins = [amount + 1] * (amount + 1)
        minCoins[0] = 0
        
        
        for i in range(amount + 1):
            
            for coin in coins:
                
                if coin <= i:
                    
                    
                    
                    minCoins[i] = min(minCoins[i], minCoins[i-coin] + 1)
        
        
        if minCoins[amount] == amount + 1:
            return -1
        
        
        return minCoins[amount]
                
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        dp = [1]*(len(nums))
        output = 1
        for i in range(len(nums)):
            for j in range(i-1,-1,-1):
                if nums[i] > nums[j]:
                    dp[i] = max(dp[j] + 1,dp[i])
                    output = max(output,dp[i])
        return output


class Solution:     
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    

                    

                    

                    
                    
                    
                    
                    
                    

                    
                    
                    
                                
    def lengthOfLIS(self, nums: list[int]) -> int:

        arr = [nums.pop(0)]                  
 
        for n in nums:                       
            
            if n > arr[-1]:                  
                arr.append(n)

            else:                            
                arr[bisect.bisect_left(arr, n)] = n 

        return len(arr)                      


class Solution:     
    def lengthOfLIS(self, nums: List[int]) -> int:
        tails = [0] * len(nums)
        result = 0
        for num in nums:
            left_index, right_index = 0, result
            while left_index != right_index:
                middle_index = left_index + (right_index - left_index) // 2
                if tails[middle_index] < num:
                    left_index = middle_index + 1
                else:
                    right_index = middle_index
            result = max(result, left_index + 1)
            tails[left_index] = num
        return result
    


class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        res = []

        for num in nums:
            if not res or num > res[-1]:
                res.append(num)
            else:
                idx = bisect.bisect_left(res,num)
                res[idx] = num

        return len(res)

class Solution:
    def minimumTotal(self, triangle: List[List[int]]) -> int:
        dp = triangle[-1]
        for i in range(len(triangle)-2,-1,-1):
            for j in range(len(triangle[i])):
                dp[j] = triangle[i][j] + min(dp[j], dp[j+1])
        return dp[0]
        
class Solution:
    def minimumTotal(self, triangle: List[List[int]]) -> int:
        for i in range(len(triangle)-2, -1, -1): 
            for j in range(i+1):                
                triangle[i][j] += min(triangle[i+1][j],    
                                      triangle[i+1][j+1]) 
        return triangle[0][0]

class Solution:
    def minimumTotal(self, triangle: List[List[int]]) -> int:
        for i in range(1, len(triangle)):  
            for j in range(i+1):           
                triangle[i][j] += min(triangle[i-1][j-(j==i)],  
                                      triangle[i-1][j-(j>0)])   
        return min(triangle[-1])  


class Solution:
    def minimumTotal(self, a: List[List[int]]) -> int:
        def dfs(level, i):            
            return 0 if level >= len(a) else a[level][i] + min(dfs(level + 1, i), dfs(level + 1, i+1))        
        return dfs(0, 0)            

class Solution:
    def minPathSum(self, grid: List[List[int]]) -> int:
        dp = grid
        for i in range(len(grid)):
            for j in range(len(grid[i])):
                if i == j and i == 0:
                    continue
                above = left = None
                if i-1 >= 0:
                    above = dp[i-1][j]
                if j-1 >= 0:
                    left = dp[i][j-1]
                if above and left:
                    dp[i][j] += min(above,left) 
                else:
                    dp[i][j] += left or above
        return dp[-1][-1]
        
class Solution:
    def minPathSum(self, grid: List[List[int]]) -> int:
        for i in range(len(grid)):
            for j in range(len(grid[i])):
                if i == 0 and j == 0:
                    
                    continue
                above = float('inf') if i-1 < 0 else grid[i-1][j]
                left = float('inf') if j-1 < 0 else grid[i][j-1]
                grid[i][j] += min(above, left)
        return grid[-1][-1]

class Solution:
    def minPathSum(self, grid: List[List[int]]) -> int:
            
        
        m, n = len(grid), len(grid[0])
        
        for i in range(1, m):
            grid[i][0] += grid[i-1][0]
        
        for i in range(1, n):
            grid[0][i] += grid[0][i-1]
        
        for i in range(1, m):
            for j in range(1, n):
                grid[i][j] += min(grid[i-1][j], grid[i][j-1])
        
        return grid[-1][-1]
    
        


class Solution:
    def uniquePathsWithObstacles(self, obstacleGrid: List[List[int]]) -> int:
        if not obstacleGrid or not obstacleGrid[0] or obstacleGrid[0][0] == 1:
            return 0
        for i in range(len(obstacleGrid)):
            for j in range(len(obstacleGrid[i])):
                if obstacleGrid[i][j] == 1 or (i == 0 and j ==0):
                    obstacleGrid[i][j] ^= 1
                    continue
                above = 0 if i-1 < 0 else obstacleGrid[i-1][j]
                left = 0 if j-1 < 0 else obstacleGrid[i][j-1]
                obstacleGrid[i][j] += above + left
        return obstacleGrid[-1][-1]
        
from functools import lru_cache
class Solution:
    def uniquePathsWithObstacles(self, obstacleGrid: List[List[int]]) -> int:
        M, N = len(obstacleGrid), len(obstacleGrid[0])
        
        @lru_cache(maxsize=None)
        def dfs(i, j):
            if obstacleGrid[i][j]:      
                return 0
            if i == M-1 and j == N-1:   
                return 1
            count = 0
            if i < M-1:
                count += dfs(i+1, j)    
            if j < N-1:
                count += dfs(i, j+1)    
            return count
        
        return dfs(0, 0)

from functools import lru_cache
class Solution:
    def uniquePathsWithObstacles(self, obstacleGrid: List[List[int]]) -> int:
        M, N = len(obstacleGrid), len(obstacleGrid[0])
        
        @lru_cache(maxsize=None)
        def dfs(i, j):
            if obstacleGrid[i][j]:      
                return 0
            if i == M-1 and j == N-1:   
                return 1
            count = 0
            if i < M-1:
                count += dfs(i+1, j)    
            if j < N-1:
                count += dfs(i, j+1)    
            return count
        
        return dfs(0, 0)

class Solution:
    def uniquePathsWithObstacles(self, obstacleGrid: List[List[int]]) -> int:              
        m, n = len(obstacleGrid), len(obstacleGrid[0])        
        
        dp=[[0] * (n+1) for _ in range(m+1)]        
        dp[0][1]=1
                        
        for row in range(1, m+1):
            for col in range(1, n+1):
                if not obstacleGrid[row-1][col-1]:
                    dp[row][col] = dp[row-1][col] + dp[row][col-1]
         
        return dp[-1][-1]


class Solution:
    def longestPalindrome(self, s: str) -> str:
        output1 = 0
        output2 = 0
        string_length = len(s)
        dp=[[0] * (string_length) for i in range(string_length)]  

        for i in range(string_length):
            dp[i][i] = 1
            if i+1 < string_length and s[i] == s[i+1]:
                dp[i][i+1] = 1
                output1 = i
                output2 = i+1

        for i in range(2,string_length):
            for j in range(string_length-i):
                if s[j] == s[j+i] and dp[j+1][j+i-1]:
                    dp[j][i+j] = 1
                    output1 = j
                    output2 = j+i
        return s[output1:output2+1]
        

        
        

class Solution:
    def longestPalindrome(self, s):
        longest_palindrom = ''
        dp = [[0]*len(s) for _ in range(len(s))]
        
        for i in range(len(s)):
            dp[i][i] = True
            longest_palindrom = s[i]
			
        
        for i in range(len(s)-1,-1,-1):
				
            for j in range(i+1,len(s)):  
                if s[i] == s[j]:  
                    
                    
                    if j-i ==1 or dp[i+1][j-1] is True:
                        dp[i][j] = True
                        
                        if len(longest_palindrom) < len(s[i:j+1]):
                            longest_palindrom = s[i:j+1]
                
        return longest_palindrom


class Solution:
    def longestPalindrome(self, s: str) -> str:
        if len(s) <= 1:
            return s

        def expand_from_center(left, right):
            while left >= 0 and right < len(s) and s[left] == s[right]:
                left -= 1
                right += 1
            return s[left + 1:right]

        max_str = s[0]

        for i in range(len(s) - 1):
            odd = expand_from_center(i, i)
            even = expand_from_center(i, i + 1)

            if len(odd) > len(max_str):
                max_str = odd
            if len(even) > len(max_str):
                max_str = even

        return max_str


class Solution:
    def longestPalindrome(self, s: str) -> str:
        if len(s) <= 1:
            return s
        
        Max_Len=1
        Max_Str=s[0]
        s = '#' + '#'.join(s) + '#'
        dp = [0 for _ in range(len(s))]
        center = 0
        right = 0
        for i in range(len(s)):
            if i < right:
                dp[i] = min(right-i, dp[2*center-i])
            while i-dp[i]-1 >= 0 and i+dp[i]+1 < len(s) and s[i-dp[i]-1] == s[i+dp[i]+1]:
                dp[i] += 1
            if i+dp[i] > right:
                center = i
                right = i+dp[i]
            if dp[i] > Max_Len:
                Max_Len = dp[i]
                Max_Str = s[i-dp[i]:i+dp[i]+1].replace('#','')
        return Max_Str


class Solution:
    def isInterleave(self, s1: str, s2: str, s3: str) -> bool:
        if len(s1) + len(s2) != len(s3):
            return False
        dp = [[0]*(len(s2)+1) for _ in range(len(s1)+1)]
        dp[0][0] = 1
        for i in range(len(s1)+1):
            for j in range(len(s2)+1):
                if i > 0 and dp[i-1][j] and s1[i-1] == s3[j+i-1]:
                    dp[i][j] = 1
                elif j > 0 and  dp[i][j-1] and s2[j-1] == s3[j+i-1]:
                    dp[i][j] = 1
        return dp[-1][-1] == 1

                


class Solution:
    def isInterleave(self, s1: str, s2: str, s3: str) -> bool:
        m, n, l = len(s1), len(s2), len(s3)
        if m + n != l:
            return False
        
        dp = [[False] * (n + 1) for _ in range(m + 1)]
        dp[0][0] = True
        
        for i in range(1, m + 1):
            dp[i][0] = dp[i-1][0] and s1[i-1] == s3[i-1]
        
        for j in range(1, n + 1):
            dp[0][j] = dp[0][j-1] and s2[j-1] == s3[j-1]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                dp[i][j] = (dp[i-1][j] and s1[i-1] == s3[i+j-1]) or (dp[i][j-1] and s2[j-1] == s3[i+j-1])
        
        return dp[m][n]


class Solution:
    def isInterleave(self, s1: str, s2: str, s3: str) -> bool:
        m, n, l = len(s1), len(s2), len(s3)
        if m + n != l:
            return False
        
        if m < n:
            return self.isInterleave(s2, s1, s3)
        
        dp = [False] * (n + 1)
        dp[0] = True
        
        for j in range(1, n + 1):
            dp[j] = dp[j-1] and s2[j-1] == s3[j-1]
        
        for i in range(1, m + 1):
            dp[0] = dp[0] and s1[i-1] == s3[i-1]
            for j in range(1, n + 1):
                dp[j] = (dp[j] and s1[i-1] == s3[i+j-1]) or (dp[j-1] and s2[j-1] == s3[i+j-1])
        
        return dp[n]


class Solution:
    def isInterleave(self, s1: str, s2: str, s3: str) -> bool:
        m, n, l = len(s1), len(s2), len(s3)
        if m + n != l:
            return False
        
        memo = {} 
        
        def helper(i: int, j: int, k: int) -> bool:
            if k == l:
                return True
            
            if (i, j) in memo:
                return memo[(i, j)]
            
            ans = False
            if i < m and s1[i] == s3[k]:
                ans = ans or helper(i + 1, j, k + 1)
                
            if j < n and s2[j] == s3[k]:
                ans = ans or helper(i, j + 1, k + 1)
            
            memo[(i, j)] = ans
            return ans
        
        return helper(0, 0, 0)





class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        m, n = len(word1), len(word2)
        
        dp = [0] * (n+1)
        
        for j in range(1, n+1):
            dp[j] = dp[j-1] + 1
        
        for i in range(1,m+1):
            previous_diag = dp[0]
            dp[0] = dp[0] + 1
            for j in range(1,n+1): 
                if word1[i - 1] == word2[j - 1]:
                    dp[j], previous_diag = previous_diag, dp[j]  
                else:
                    dp[j], previous_diag = 1 + min(
                        dp[j],    
                        dp[j - 1],    
                        previous_diag 
                ), dp[j]

        return dp[n]
        

def minDistance(self, word1: str, word2: str) -> int:
    m, n = len(word1), len(word2)
    
    
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    
    for j in range(1, n + 1):
        dp[0][j] = j  
    
    
    for i in range(1, m + 1):
        dp[i][0] = i  
    
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i - 1] == word2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]  
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],    
                    dp[i][j - 1],    
                    dp[i - 1][j - 1] 
                )
    
    
    return dp[m][n]


class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        n = len(prices)
        max_profit_one_transaction = [0]*n
        min_price = prices[0]
        
        for i in range(1,n):
            max_profit_one_transaction[i] = max(max_profit_one_transaction[i-1], prices[i]-min_price)
            min_price = min(min_price, prices[i])

        max_profit = 0
        max_price = prices[-1]
        print(max_profit_one_transaction)
        for i in range(n-2,-1,-1):
            print(max_profit)
            max_profit = max(max_profit, max_profit_one_transaction[i]-prices[i]+max_price)
            max_price = max(max_price, prices[i])
        return max_profit


class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        
        
        dp_2_hold, dp_2_not_hold = -float('inf'), 0
        dp_1_hold, dp_1_not_hold = -float('inf'), 0
        
        for stock_price in prices:
            
            
            dp_2_not_hold = max( dp_2_not_hold, dp_2_hold + stock_price )
            
            
            dp_2_hold = max( dp_2_hold, dp_1_not_hold - stock_price )
            
            
            dp_1_not_hold = max( dp_1_not_hold, dp_1_hold + stock_price )
            
            
            dp_1_hold = max( dp_1_hold, 0 - stock_price )
            
        
        return dp_2_not_hold

class Solution:
  def maxProfit(self, prices: List[int]) -> int:
    if not prices:
        return 0

    
    buy1, buy2 = float('inf'), float('inf')
    sell1, sell2 = 0, 0

    
    for price in prices:
        
        buy1 = min(buy1, price)
        sell1 = max(sell1, price - buy1)
        
        buy2 = min(buy2, price - sell1)
        sell2 = max(sell2, price - buy2)

    return sell2


class Solution:
    def maxProfit(self,prices):
        if not prices:
            return 0
        
        
        
        dp = [[[0 for _ in range(2)] for _ in range(3)] for _ in range(len(prices))]
        
        
        dp[0][0][0], dp[0][0][1] = 0, -prices[0]
        dp[0][1][0], dp[0][1][1] = 0, -prices[0]
        dp[0][2][0], dp[0][2][1] = 0, -prices[0]
        
        for i in range(1, len(prices)):
            for j in range(1, 3):
                
                dp[i][j][0] = max(dp[i-1][j][0], dp[i-1][j][1] + prices[i])
                
                dp[i][j][1] = max(dp[i-1][j][1], dp[i-1][j-1][0] - prices[i])
        
        
        return dp[-1][2][0]


class Solution:
    def maxProfit(self, k: int, prices: List[int]) -> int:
        if not prices:
            return 0
        
        
        
        dp = [[[0 for _ in range(2)] for _ in range(k+1)] for _ in range(len(prices))]
        
        for i in range(k+1):
            for j in range(2):
                if j:
                    dp[0][i][j] = -prices[0]
                else:
                    dp[0][i][j] = 0
        
        for i in range(1, len(prices)):
            for j in range(1, k+1):
                
                dp[i][j][0] = max(dp[i-1][j][0], dp[i-1][j][1] + prices[i])
                
                dp[i][j][1] = max(dp[i-1][j][1], dp[i-1][j-1][0] - prices[i])
        
        
        return dp[-1][k][0]


class Solution:
    def maxProfit(self, k: int, prices: List[int]) -> int:
        
        if k == 0: return 0
        
        
        dp = [[1000, 0] for _ in range(k + 1)]
        for price in prices:
            for i in range(1, k + 1):
                
                
                
                dp[i][0] = min(dp[i][0], price - dp[i - 1][1])
                
                
                dp[i][1] = max(dp[i][1], price - dp[i][0])
        
		
        return dp[k][1]

class Solution:
    def maxProfit(self, k: int, prices: List[int]) -> int:
        if k >= len(prices)//2: return sum(max(0, prices[i] - prices[i-1]) for i in range(1, len(prices)))
        buy, sell = [inf]*k, [0]*k
        for x in prices:
            for i in range(k):
                if i: buy[i] = min(buy[i], x - sell[i-1])
                else: buy[i] = min(buy[i], x)
                sell[i] = max(sell[i], x - buy[i])
        return sell[-1] if k and prices else 0

class Solution:
    def maxProfit(self, k: int, prices: List[int]) -> int:
        if k >= len(prices)//2: return sum(max(0, prices[i] - prices[i-1]) for i in range(1, len(prices)))
        ans = [0]*len(prices)
        for _ in range(k):
            most = 0
            for i in range(1, len(prices)):
                most = max(ans[i], most + prices[i] - prices[i-1])
                ans[i] = max(ans[i-1], most)
        return ans[-1]

class DoubleLinkListNode:
    
    def __init__(self, ind, pre=None, next=None):
        self.ind = ind  
        self.pre = pre if pre else self  
        self.next = next if next else self  

class Solution:
    
    def MinMaxList(self, arr: List[int]) -> List[int]:
        n = len(arr)  
        if n == 0:  
            return []
        sign = -1  
        res = [9999]  
        for num in arr:
            
            if num * sign > res[-1] * sign:
                res[-1] = num
            else:
                
                res.append(num)
                sign *= -1
        
        if len(res) & 1:
            res.pop()
        return res

    
    def maxProfit(self, k: int, prices: List[int]) -> int:
        newP = self.MinMaxList(prices)  
        n = len(newP)  
        m = n // 2  
        res = 0  
        
        for i in range(m):
            res += newP[i*2+1] - newP[i*2]
        
        if m <= k:
            return res

        
        head, tail = DoubleLinkListNode(-1), DoubleLinkListNode(-1)
        NodeList = [DoubleLinkListNode(0, head)]
        for i in range(1, n):
            NodeList.append(DoubleLinkListNode(i, NodeList[-1]))
            NodeList[i-1].next = NodeList[i]
        NodeList[n-1].next = tail
        head.next, tail.pre = NodeList[0], NodeList[n-1]

        
        heap = []
        for i in range(n-1):
            if i & 1:
                heapq.heappush(heap, [newP[i] - newP[i+1], i, i+1, 0])
            else:
                heapq.heappush(heap, [newP[i+1] - newP[i], i, i+1, 1])

        
        while m > k:
            loss, i, j, t = heapq.heappop(heap)
            if NodeList[i] is None or NodeList[j] is None:
                continue
            m -= 1
            res -= loss
            nodei, nodej = NodeList[i], NodeList[j]
            nodel, noder = nodei.pre, nodej.next
            l, r = nodel.ind, noder.ind
            valL, valR = newP[l], newP[r]
            noder.pre, nodel.next = nodel, noder
            NodeList[i], NodeList[j] = None, None
            if t == 0:
                heapq.heappush(heap, [valR - valL, l, r, 1])
            elif l != -1 and r != -1:
                heapq.heappush(heap, [valL - valR, l, r, 0])

        return res

class Solution:
    def maximalSquare(self, matrix: List[List[str]]) -> int:
        m, n = len(matrix), len(matrix[0])
        
        dp = [0] * (n)
        largest = 0

        for j in range(0,n):
            dp[j] = int(matrix[0][j])
            largest = max(largest,dp[j])
        
        for i in range(1,m):
            previous_diag = dp[0]
            dp[0] = int(matrix[i][0])
            largest = max(largest,dp[0])
            for j in range(1,n): 
                if matrix[i][j] == '1':
                    dp[j], previous_diag = (min(dp[j-1],dp[j],previous_diag) + 1), dp[j]
                    largest = max(largest,dp[j])
                else:
                    dp[j], previous_diag = 0, dp[j]
                
        return largest*largest
        
def maximalSquare(self, matrix: List[List[str]]) -> int:
        m, n = len(matrix), len(matrix[0])
        
        
        dp = [[0 for _ in range(n)] for _ in range(m)]
        largest = 0

        
        for j in range(n):
            dp[0][j] = int(matrix[0][j])
            largest = max(largest, dp[0][j])
        
        
        for i in range(m):
            dp[i][0] = int(matrix[i][0])
            largest = max(largest, dp[i][0])

        
        for i in range(1, m):
            for j in range(1, n):
                if matrix[i][j] == '1':
                    
                    dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
                    largest = max(largest, dp[i][j])
        
        return largest * largest  

class Solution:
    def maximalSquare(self, matrix):
        max_square = 0
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                if matrix[i][j] == "0":
                    continue
                if i > 0 and j > 0:
                    matrix[i][j] = str(min(int(matrix[i-1][j]), 
                                           int(matrix[i][j-1]), 
                                           int(matrix[i-1][j-1])) + 1)
                max_square = max(int(matrix[i][j]), max_square)
        
        return max_square ** 2
