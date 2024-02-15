var merge = function(nums1, m, nums2, n) {
    i = m - 1
    j = n - 1
    k = m + n - 1
    
    while (j >= 0) {
        if (i >= 0 && nums1[i] > nums2[j]){
            nums1[k] = nums1[i];
            i = i - 1
            k = k - 1
            }
        else{
            nums1[k] = nums2[j];
            j = j - 1
            k = k - 1}
    }
};

var merge = function(nums1, m, nums2, n) {
    for (let i = m, j = 0; j < n; i++, j++) {
        nums1[i] = nums2[j];
    }
    nums1.sort((a, b) => a - b);
};

var removeElement = function(nums, val) {
    i = 0
    j = 0
    while (i <= nums.length-1){
        if (nums[i] != val){
            nums[j] = nums[i]
            j += 1
            i += 1}
        else{
            i += 1}}
    return j
};

var removeElement = function(nums, val) {
    var zeroStartIdx = 0;
    for(let i=0;i<nums.length;i++){
        if(nums[i]!==val){
            nums[zeroStartIdx]=nums[i];
            zeroStartIdx++
        }
    }
    return zeroStartIdx; 
};

var removeDuplicates = function(nums) {
    var j = 1
    for (var i = 1; i < nums.length; i++){
        if (nums[i] !== nums[i - 1]){
            nums[j] = nums[i]
            j += 1}
    }
return j;
};

var removeDuplicates = function(nums) {
    var j = 1
    for (var i = 2; i < nums.length; i++){
        if (nums[i] !== nums[j - 1]){
            nums[++j] = nums[i]
            }
    }
return j+1;
};

var removeDuplicates = function(nums) {
    
    if(nums.length <= 2) {
        return nums.length;
    }
    
    
    let k = 2;
    
    for(let i = 2; i < nums.length; i++){
        
        if(nums[i] != nums[k - 2]){
            nums[k] = nums[i];
            k++;
        
        }
    }
    return k;       
};

var majorityElement = function(nums) {
    nums.sort(function(a,b) {
        return a-b
    })
    return(nums[Math.floor(nums.length/2)])
    
};


var majorityElement = function(nums) {
    
    let sol = 0, cnt = 0;
    
    for(let i = 0; i < nums.length; i++ ) {
        
        if(cnt == 0){
            sol = nums[i];
            cnt = 1;
        }
        
        else if(sol == nums[i]){
            cnt++;
        }
        
        else{
            cnt--;
        }
    }
    
    return sol;
};

var rotate = function(nums, k) {
    for( var i = 0; i < k%nums.length; i++){
        a=nums.pop()
        nums.splice(0,0,a)
}
}

var rotate = function(nums, k) {
    k = k % nums.length;
    let removed = nums.splice(-k);
    nums.unshift(...removed);
}

var rotate = function(nums, k) {
    n = nums.length
    k = k % n;
    function reverse(nums, start, end) {
        while (start < end) {
            let temp = nums[start];
            nums[start] = nums[end];
            nums[end] = temp;
            start++;
            end--;
        }}
    reverse(nums, 0, n-k-1);
    reverse(nums, n-k,n-1)
    reverse(nums,0,n-1)
}

var maxProfit = function(prices) {
    if(prices == null || prices.length <= 1) return 0;
    let minBuy = prices[0];
    let profit = 0;
    for(let i = 1; i < prices.length; i++) {
        minBuy = Math.min(minBuy, prices[i]);
        profit = Math.max(profit, prices[i] - minBuy);
    }
    return profit;
};

var maxProfit = function(prices) {
    let profit_from_price_gain = 0
    for (let i = 0; i < prices.length - 1; i++){       
        if (prices[i] < prices[i+1]){
            profit_from_price_gain += ( prices[i+1] - prices[i])
        }
    }
    return profit_from_price_gain
};

var canJump = function(nums) {
    let PossibleJumps = nums[0]
    for (let i = 1; i < nums.length; i++) {
        if (PossibleJumps == 0){
            return false}
        PossibleJumps -= 1
        PossibleJumps = Math.max(PossibleJumps, nums[i])}
    return true;
    
};


var jump = function(nums) {

    let ans = 0
    let end = 0
    let farthest = 0
    for(let i = 0; i < nums.length -1; i++){
      farthest = Math.max(farthest, i + nums[i])
      if (farthest >= nums.length - 1){
        ans += 1
        break}
      if (i == end){   
        ans += 1       
        end = farthest }
    }
    return ans
};

var hIndex = function(citations) {
    citations.sort(function(a,b) {
        return a-b
    })
    let current = 0
    for(let i = 0; i < citations.length; i++){
        if(citations[i] > citations.length - i){
            return Math.max(current, (citations.length - i))}
        current = citations[i]}
    return current
};


var RandomizedSet = function() {
    this.map = new Map();
    this.list = [];
};

RandomizedSet.prototype.insert = function(val) {
    if (this.map.has(val)) return false;

    this.map.set(val, this.list.length);
    this.list.push(val);
    return true;
};

RandomizedSet.prototype.remove = function(val) {
    if (!this.map.has(val)) return false;
    const idx = this.map.get(val);
    this._swap(idx, this.list.length - 1);
    this.list.pop();
    this.map.set(this.list[idx], idx);
    this.map.delete(val);
    return true;
};

RandomizedSet.prototype.getRandom = function() {
    return this.list[Math.floor(Math.random() * this.list.length)];
};

RandomizedSet.prototype._swap = function(a, b) {
    const tmp = this.list[a];
    this.list[a] = this.list[b];
    this.list[b] = tmp;
};


class RandomizedSet {
    constructor() {
        this.map = new Map();
        this.list = [];
    }

        insert(val) {
        if (this.map.has(val)) return false;

        this.map.set(val, this.list.length);
        this.list.push(val);
        return true;
    }

        remove(val) {
        if (!this.map.has(val)) return false;
        const idx = this.map.get(val);
        this._swap(idx, this.list.length - 1);
        this.list.pop();
        this.map.set(this.list[idx], idx);
        this.map.delete(val);
        return true;
    }

        getRandom() {
        return this.list[Math.floor(Math.random() * this.list.length)];
    }

    _swap(a,b) {
        const tmp = this.list[a];
        this.list[a] = this.list[b];
        this.list[b] = tmp;
    };
    
    }

var productExceptSelf = function(nums) {
    let length=nums.length
    let sol = new Array(length).fill(1);
    let pre = 1
    let post = 1
    for(let i = 0; i <length; i++){
        sol[i] *= pre
        pre = pre*nums[i]
        sol[length-i-1] *= post
        post = post*nums[length-i-1]}
    return(sol)
};

var canCompleteCircuit = function(gas, cost) {
    let Tank = 0
    let Total = 0
    let j = 0
    let n = gas.length
    for( let i = 0; i < n; i++){
        Total += gas[i] - cost [i]
        Tank += gas[i] - cost [i]
        if (Tank < 0){
            j = i+1
            Tank = 0}}
    return Total<0? -1:j

};

var candy = function(ratings) {
    let length = ratings.length
    let minimum = new Array(length).fill(1);
    for(let i = 1; i < length; i++){
        if (ratings[i] > ratings[i-1]){
            minimum[i] += minimum[i-1] }}
    for(let j = length -2; j > -1; j--){
        if (ratings[j] > ratings[j+1] && minimum[j] <= minimum[j+1]){
            minimum[j] = minimum[j+1] + 1}}
    return minimum.reduce((total, value) => total + value, 0);

};

var romanToInt = function(s) {
    let translations = new Map([
        ["I", 1],
        ["V", 5],
        ["X", 10],
        ["L", 50],
        ["C", 100],
        ["D", 500],
        ["M", 1000],
    ])
    number = 0
    s = s.replace("IV", "IIII").replace("IX", "VIIII")
    s = s.replace("XL", "XXXX").replace("XC", "LXXXX")
    s = s.replace("CD", "CCCC").replace("CM", "DCCCC")
    for (let char of s){
        number += translations.get(char)}
    return number
};

var romanToInt = function(s) {
    let symbols = new Map([
        ["I", 1],
        ["V", 5],
        ["X", 10],
        ["L", 50],
        ["C", 100],
        ["D", 500],
        ["M", 1000],
    ])
    number = 0
    for(let i = 0; i < s.length; i+=1){
        (s[i+1] !== undefined && symbols.get(s[i]) < symbols.get(s[i+1])) ? number -= symbols.get(s[i]) : number += symbols.get(s[i]);
    }
    return number
    };


var romanToInt = function(s) {
    let symbols = {
        "I": 1,
        "V": 5,
        "X": 10,
        "L": 50,
        "C": 100,
        "D": 500,
        "M": 1000
    };
    let value = 0;
    for(let i = 0; i < s.length; i+=1){
        (s[i+1] && symbols[s[i]] < symbols[s[i+1]]) ? value -= symbols[s[i]]: value += symbols[s[i]]
    }
    return value
    };

var trap = function(height) {
    if (height.length <= 2){
        return 0}
    
    let ans = 0
    let i = 1
    let j = height.length - 1
    
    let lmax = height[0]
    let rmax = height[j]
    
    while(i <=j){

        if (height[i] > lmax){
            lmax = height[i]}
        if (height[j] > rmax){
            rmax = height[j]}
        
        if (lmax <= rmax){
            ans += lmax - height[i]
            i += 1}
            
        else{
            ans += rmax - height[j]
            j -= 1}
    }
    return ans

};

var intToRoman = function(num) {
    const val = [1000,900,500,400,100,90,50,40,10,9,5,4,1]
    const rom = ["M","CM","D","CD","C","XC","L","XL","X","IX","V","IV","I"]
    let ans = ""
    for (let i = 0; num; i++)
        while (num >= val[i]) ans += rom[i], num -= val[i]
    return ans

};

var intToRoman = function(num) {
    let numerals = {'M':1000,'CM': 900, 'D': 500,'CD': 400, 'C': 100, 'XC':90 ,'L': 50, 'XL':40, 'X': 10, 'IX': 9, 'V': 5, 'IV':4,'I': 1}
    let Strings = ''
    for (key in numerals){
        if (num - numerals[key] >= 0){
            n = Math.floor(num/numerals[key])
            num -= numerals[key]*n
            Strings += key.repeat(n)}}
    return Strings
}

var lengthOfLastWord = function(s) {
    wordlist = s.trim().split(/\s+/)
    if (wordlist){
        return wordlist[wordlist.length-1].length}
    return 0
};

var longestCommonPrefix = function(strs) {
    let ans=""
    strs.sort((a, b) => a.toLowerCase().localeCompare(b.toLowerCase()));
    let first=strs[0]
    let last=strs[strs.length - 1]
    for( let  i = 0; i < Math.min(first.length,last.length);i++) {
        if(first[i]!=last[i]){
            return ans}
        ans+=first[i]}
    return ans

};

var longestCommonPrefix = function(strs) {
    return strs.reduce((prev, next) => {
        let i = 0;
        while (prev[i] && next[i] && prev[i] === next[i]) i++;
        return prev.slice(0, i);
    });
};

var reverseWords = function(s) {
    wordlist = s.trim().split(/\s+/)
    function reverse(nums, start, end) {
        while (start < end) {
            let temp = nums[start];
            nums[start] = nums[end];
            nums[end] = temp;
            start++;
            end--;
    }}
    reverse(wordlist, 0, wordlist.length -1);
    return(wordlist.join(' '))
};

function reverseWords(s) {
    const ret = [];
    let word = [];
    for (let i = 0; i < s.length; ++i) {
      if (s.charAt(i) === ' ') {
          
          word.length > 0 && ret.unshift(word.join(''));
          
          word = [];
        }
      else {
        
        word.push(s.charAt(i));
      }
    }
    
    word.length > 0 && ret.unshift(word.join(''));
    return ret.join(' ');
  };

  
  
var convert = function(s, numRows) {
    if (numRows == 1){
        return s}
    let Row = 0
    let Concat = new Array(numRows).fill('');
    let increase = 1
     for (let i of s){
        Concat[Row] += i
        Row += increase
        if (Row == numRows -1) {
            increase = -1}
        if (Row == 0){
            increase = 1}}
    return(Concat.join(''))
};

var strStr = function(haystack, needle) {
    return haystack.indexOf(needle)
};

var strStr = function(haystack, needle) {
    if (needle === '' || needle === haystack) return 0;    
    if (haystack.length < needle.length) return -1;        
    
    for (let i = 0; i < haystack.length - needle.length + 1; i++) {    
        if (haystack[i] === needle[0]) {                
        for (let j = 0; j < needle.length; j++) {     
            if (needle[j] !== haystack[i + j]) {        
            break;                                    
            } else if (j === needle.length - 1){        
            return i;                                 
            }
        }
        }
    }
    
    return -1; 
};

var fullJustify = function(words, maxWidth) {
    let output = [], line = [], letters = 0
    for(let w of words){
        if (letters + w.length + line.length > maxWidth){
            for( let i = 0; i < maxWidth - letters; i++){
                line[i%(line.length-1 || 1)] += " "}
            output.push(line.join(''))
            line = [], letters = 0}
        line = line.concat([w])
        letters += w.length}
    string = line.join(' ')
    string +=' '.repeat(maxWidth - string.length)
    return output.concat([string])

};

var fullJustify = function(words, maxWidth) {
    let result = [];
    
    let line = [];
    let lineLength = 0;
    
    for(let i = 0; i < words.length; i++) {
        let w = words[i];
        
        if(lineLength === 0 && w.length <= maxWidth) {
			
			
			
            line.push(w);
            lineLength += w.length;
        } else if(lineLength + w.length + 1 <= maxWidth){
			
            line.push(w);
            lineLength += (w.length + 1);
        } else {
			
			
            
            line = addMinSpace(line);
            
            
            let remainingSpace = maxWidth - lineLength;
            
            
            line = distributeSpaces(line, remainingSpace);

            
            let temp = line.join("")
            
            
            if(line.length === 1) temp = addRemainingSpaces(temp, remainingSpace)
            
            result.push(temp);
            
            
            line = [];
            lineLength = 0;
            
            
            line.push(w);
            lineLength += w.length;
        }
    }
    
    
    
    line = addMinSpace(line);
    
    
    let temp = line.join("")
    
    
    let remainingSpace = maxWidth - lineLength;
    
    
    temp = addRemainingSpaces(temp, remainingSpace)
    
    
    result.push(temp);
 
    
    return result;
    
	
    function addMinSpace(line) {
        for(let i = 0; i < line.length - 1; i++) line[i] += " ";
        return line;
    }
    
	
    function addRemainingSpaces(line, spaces) {
        while(spaces > 0) {
            line += " ";
            spaces--;
        }
        return line;
    }
    
	
    function distributeSpaces(arr, spaces) {
        while(spaces > 0 && arr.length > 1) {
           for(let i = 0; i < arr.length - 1; i++) {
                if(spaces <= 0) break;
                arr[i] = arr[i] + " ";
                spaces --;
            } 
        }
        return arr;
    }
};

function isPalindrome(s) {
    s = Array.from(s.toLowerCase()).filter(char => /[a-z0-9]/i.test(char)).join('');
    return s === s.split('').reverse().join('');
}


var isPalindrome = function(s) {
    s = s.toLowerCase().replace(/[^a-z0-9]/gi,'');
    for (let i = 0, j = s.length - 1; i <= j; i++, j--) {
    if (s.charAt(i) !== s.charAt(j)) return false;
    }
    return true;
};

var isSubsequence = function(s, t) {
    for (let c of s){
        i = t.indexOf(c)
        if(i == -1) return false;
        else t = t.substring(i+1)}
    return true
};

var twoSum = function(numbers, target) {
    let i = 0
    let j = numbers.length - 1
    while (i < j){
        if(numbers[j] + numbers[i] == target){
            return [i+1,j+1]}
        else if(numbers[j] + numbers[i] > target){
            j -= 1}
        else{
            i += 1}}
};


var maxArea = function(height) {
    let i = 0
    let j = height.length -1
    let maxwater = 0
    while (i <j){
        maxwater = Math.max((j-i)*Math.min(height[i],height[j]),maxwater)
        if (height[i] > height[j]){
            j -= 1}
        else{
            i +=1}}
    return maxwater

};

var maxArea = function(H) {
    let ans = 0, i = 0, j = H.length-1
    while (i < j) {
        ans = Math.max(ans, Math.min(H[i], H[j]) * (j - i))
        H[i] <= H[j] ? i++ : j--
    }
    return ans
};

var threeSum = function(nums) {
    let res = new Set()
    let n = [], p = [], z = []
    let Neg = new Set(), Pos = new Set()

    for(let num of nums){
        if (num > 0){
            p.push(num)
            Pos.add(num)}
        else if( num < 0){
            n.push(num)
            Neg.add(num)}
        else{
            z.push(num)}}


    if (z.length > 0){
        for (let num of Pos){
            if ( Neg.has(-1*num)){
                res.add([-1*num, 0, num].toString())}}}

    if (z.length >= 3){
        res.add([0,0,0].toString())}

    for(let i = 0; i < n.length; i++){
        for(let j = i+1; j < n.length; j++){
            let target = -1*(n[i]+n[j])
            if(Pos.has(target)){
                res.add([n[i],n[j],target].sort((a, b) => a - b).toString()
)}}}

   for(let i = 0; i < p.length; i++){
        for(let j = i+1; j < p.length; j++){
            let target = -1*(p[i]+p[j])
            if(Neg.has(target)){
                res.add([p[i],p[j],target].sort((a, b) => a - b).toString()
)}}}
    return Array.from(res, str => str.split(',').map(Number));
};

var threeSum = function(nums) {
    nums.sort((a, b) => a - b);
    const result = [];
    
    for(let i = 0; i < nums.length; i++) {
        let low = i+1, high = nums.length-1, sum = 0;
        
        while(low < high) {
            sum = nums[i] + nums[low] + nums[high];
            
            if(sum === 0) {
                result.push([nums[i], nums[low], nums[high]]);
                while(nums[low+1] === nums[low]) low++;
                while(nums[high-1] === nums[high]) high--;
                low++;
                high--;
            } else if(sum < 0) low++;
            else high--;
        }
        while(nums[i+1] === nums[i]) i++;
    }
    return result;    
};

var minSubArrayLen = function(target, nums) {
    let j = 0
    let c = Infinity
    let total = 0
    for(let i = 0; i < nums.length; i++){
        total += nums[i]
        while(total >= target){
            c = Math.min(c, i - j +1)
            total -= nums[j]
            j +=1}}
    return (c != Infinity) ? c : 0;
};

var lengthOfLongestSubstring = function(s) {
    let seen = {}
    let l = 0
    let output = 0
    for(let i = 0; i < s.length; i++ ){
        if(!(s[i] in seen)){
            output = Math.max(output,i-l+1)}
        else{
            if(seen[s[i]] < l){
                output = Math.max(output,i-l+1)}
            else{
                l = seen[s[i]] + 1
            }}
        seen[s[i]] = i}
    return output

};

var lengthOfLongestSubstring = function(s) {
    let set = new Set();
    let left = 0;
    let maxSize = 0;

    if (s.length === 0) return 0;
    if (s.length === 1) return 1;

    for (let i = 0; i < s.length; i++) {

        while (set.has(s[i])) {
            set.delete(s[left])
            left++;
        }
        set.add(s[i]);
        maxSize = Math.max(maxSize, i - left + 1)
    }
    return maxSize;

};

var findSubstring = function(s, words) {
    const length = words[0].length,
        permutation_length = length * words.length
    var output = [],
        map = {}

    for( let w of words ){
        if( !map[w] ) map[w] = 1
        else map[w] += 1
    }

    
    for( let window = 0; window <= s.length - permutation_length; window++ ){
        let str_window = s.substring(window, window + permutation_length),
            words_seen = {},
            match = true
        
        for( let index = 0; index <= str_window.length - length; index += length ){ 
            let word = str_window.substring( index, index + length )
            if( map[word] && (!words_seen[word] || words_seen[word] < map[word]) ){ 
                words_seen[word] = words_seen[word] ? words_seen[word] + 1 : 1 
            } else { 
                match = false
                break 
            }
        }
        
        if( match ) output.push( window ) 
    }
    return output
};

var findSubstring = function(s, words) {
    let length = words[0].length;
    let word_count = new Map();
    words.forEach(word => {
        word_count.set(word, (word_count.get(word) || 0) + 1);
    });
    let indexes = [];

    for (let i = 0; i < length; i++) {
        let start = i;
        let window = new Map();
        let words_used = 0;

        for (let j = i; j <= s.length - length; j += length) {
            let word = s.substring(j, j + length);

            if (!word_count.has(word)) {
                start = j + length;
                window = new Map();
                words_used = 0;
                continue;
            }

            words_used++;
            window.set(word, (window.get(word) || 0) + 1);

            while (window.get(word) > word_count.get(word)) {
                let firstWord = s.substring(start, start + length);
                window.set(firstWord, window.get(firstWord) - 1);
                start += length;
                words_used--;
            }

            if (words_used === words.length) {
                indexes.push(start);
            }
        }
    }

    return indexes;
}

var minWindow = function(s, t) {
    if(!s || !t || s.length < t.length){
        return ''}
    
    let t_counter = new Map()
    for (let i = 0; i < t.length; i++) {
        let char = t[i];
        t_counter.set(char, (t_counter.get(char) || 0) + 1);
    }
    let chars = t_counter.size
    
    let s_counter = new Map()
    let matches = 0
    
    let answer = ''
    
    let i = 0
    let j = -1 
    
    while(i < s.length){
        
        if (matches < chars){
            
            if(j == s.length - 1)
                return answer
            j += 1
            s_counter.set(s[j], (s_counter.get(s[j]) || 0) + 1);
            if(t_counter.get(s[j])> 0 && s_counter.get(s[j]) == t_counter.get(s[j])){
                matches += 1}
        }

        else{
            s_counter.set(s[i], (s_counter.get(s[i])) - 1);
            if (t_counter.get(s[i]) > 0 && s_counter.get(s[i]) == t_counter.get(s[i]) - 1){
                matches -= 1}
            i += 1}

        if(matches == chars){
            if(!answer){
                answer = s.substring(i,j+1)}
            else if ((j - i + 1) < answer.length){
                answer = s.substring(i,j+1)}
    }}
    return answer    
};

var minWindow = function(s, t) {
    
  
  let min = "", left = 0, right = -1;
  let map = {};
  
  
  
  
  t.split('').forEach(element => {
      if (map[element]==null) map[element] = 1;
      else map[element] = map[element] + 1;
  });
  
  
  
  let count = Object.keys(map).length;

  while (right <= s.length) {
      
      if (count == 0) {
      
          
          
          let current = s[left];
          
          
          if (map[current] != null) map[current]++;
          
          
          if (map[current] > 0) count++;    
          
          let temp = s.substring(left, right+1)
          if (min == "") min = temp;
          else min = min.length<temp.length?min:temp;
          
          left++;
      } else {
          right++;
          let current = s[right];
          
          
          if (map[current] != null) map[current]--;
          
          if (map[current] == 0) count--;
      }
  }
  return min;
}

var isValidSudoku = function(board) {
    let res = []
    for(let i = 0; i <9; i++){
        for (let j = 0; j<9; j++){
            let element = board[i][j]
            if(element !== '.'){
                res.push(`r${i}${element}`, `c${element}${j}`, `${Math.floor(i / 3)}${Math.floor(j / 3)}${element}`)}}};    
    return res.length === new Set(res).size
};

var isValidSudoku = function(board) {
    for (let i = 0; i < 9; i++) {
      let row = new Set(),
          col = new Set(),
          box = new Set();
  
      for (let j = 0; j < 9; j++) {
        let _row = board[i][j];
        let _col = board[j][i];
        let _box = board[3*Math.floor(i/3)+Math.floor(j/3)][3*(i%3)+(j%3)]
        
        if (_row != '.') {
          if (row.has(_row)) return false;
          row.add(_row);
        }
        if (_col != '.') {
          if (col.has(_col)) return false;
          col.add(_col);
        }
        
        if (_box != '.') {
          if (box.has(_box)) return false;
          box.add(_box);
        } 
      }
    }
    return true
  };

  var spiralOrder = function(matrix) {
    let res = [];
    let row_begin = 0;
    let col_begin = 0;
    let row_end = matrix.length - 1;
    let col_end = matrix[0].length - 1;

    while (res.length < matrix[0].length * matrix.length) {
        for (let i = col_begin; i <= col_end; i++) {
            res.push(matrix[row_begin][i]);
        }
        row_begin++;

        for (let i = row_begin; i <= row_end; i++) {
            res.push(matrix[i][col_end]);
        }
        col_end--;

        if (row_begin <= row_end) {
            for (let i = col_end; i >= col_begin; i--) {
                res.push(matrix[row_end][i]);
            }
            row_end--;
        }

        if (col_begin <= col_end) {
            for (let i = row_end; i >= row_begin; i--) {
                res.push(matrix[i][col_begin]);
            }
            col_begin++;
        }
    }

    return res;

};

var rotate = function(matrix) {
    length = matrix.length
    for(let i = 0; i < length ; i++){
        matrix.push([])
        for(let j = 0; j < length ; j++){
            matrix[length+i].unshift(matrix[j][i])}}
    matrix.splice(0,length)
};

var rotate = function(matrix) {
    let n = matrix.length, depth = ~~(n / 2)
    for (let i = 0; i < depth; i++) {
        let len = n - 2 * i - 1, opp = n - 1 - i
        for (let j = 0; j < len; j++) {
            let temp = matrix[i][i+j]
            matrix[i][i+j] = matrix[opp-j][i]
            matrix[opp-j][i] = matrix[opp][opp-j]
            matrix[opp][opp-j] = matrix[i+j][opp]
            matrix[i+j][opp] = temp
        }
    }

};

var setZeroes = function(matrix) {
    let m = matrix.length
    let n = matrix[0].length
    
    let first_row_has_zero = false
    let first_col_has_zero = false
    
    for(let row = 0; row < m; row++){
        for(let col = 0; col < n; col++){
            if (matrix[row][col] == 0){
                if (row == 0){
                    first_row_has_zero = true}
                if (col == 0){
                    first_col_has_zero = true}
                matrix[row][0] = matrix[0][col] = 0}}}

    for(let row = 1; row < m; row++){
        for(let col = 1; col < n; col++){
            if (matrix[0][col] == 0 || matrix[row][0] == 0){
                matrix[row][col] = 0}}} 
    
    if (first_row_has_zero){
        for(let col = 0; col < n; col++){
            matrix[0][col] = 0}}
    
    if (first_col_has_zero){
        for(let row = 0; row < m; row++){
            matrix[row][0] = 0}}
};

var setZeroes = function(matrix) {

    var track = []
    
    
    for(var i = 0; i < matrix.length; i++){
      for(var j = 0; j < matrix[0].length; j++){
        if(matrix[i][j] === 0) track.push([i, j])                
      }
    }

    for(var i = 0; i < track.length; i++){
      var [x, y] = track[i]
      
      
      for(var j = 0; j < matrix[0].length; j++){
        matrix[x][j] = 0
      }
      
      
      for(var j = 0; j < matrix.length; j++){
        matrix[j][y] = 0
      }

    }
};

var gameOfLife = function(board) {
    let directions = [[1,0], [1,-1], [0,-1], [-1,-1], [-1,0], [-1,1], [0,1], [1,1]];
    for(let i = 0; i < board.length ; i++){
        for(let j = 0; j < board[0].length ; j++){
            let live = 0                
            for (let direction of directions) {
                let x = direction[0];
                let y = direction[1];
                if (( i + x < board.length && i + x >= 0 ) && ( j + y < board[0].length && j + y >=0 ) && Math.abs(board[i + x][j + y]) == 1){
                    live += 1}}
            if (board[i][j] == 1 && (live < 2 || live > 3)){
                board[i][j] = -1}
            if (board[i][j] == 0 && live == 3){                 
                board[i][j] = 2}}}
    for(let i = 0; i < board.length ; i++){
        for(let j = 0; j < board[0].length ; j++){
            if(board[i][j] > 0){
                board[i][j] = 1}
            else {
                board[i][j] = 0}}}

};


var gameOfLife = function(board) {
    if(board.length === 0){
        return board;
    }
    
    var checkNeighbors = function(row, col){
      var score = -board[row][col];
      var r, c;
      for(r = row - 1; r <= row + 1; r++){
          for(c = col - 1; c <= col + 1; c++){
              if(typeof board[r] !== "undefined" && typeof board[r][c] !== "undefined"){
                score += Math.abs(Math.floor(board[r][c]));
              }
          }
      }
      return score;
    };
    
    var r, c;
    for(r = 0; r < board.length; r++){
        for(c = 0; c < board[0].length; c++){
            var score = checkNeighbors(r, c);
            if(board[r][c] === 1){
                if(score < 2 || score > 3){
                    board[r][c] = -0.5;
                }
            }
            else if(board[r][c] === 0){
                if(score === 3){
                    board[r][c] = 0.5;
                }
            }
        }
    }
    
    for(r = 0; r < board.length; r++){
        for(c = 0; c < board[0].length; c++){
            board[r][c] = Math.ceil(board[r][c]);
        }
    }
};

var canConstruct = function(ransomNote, magazine) {
    let mag_count = new Map();
    for (let letter of magazine) {
        mag_count.set(letter, (mag_count.get(letter) || 0) + 1);
    }

    let ran_count = new Map();
    for (let letter of ransomNote) {
        ran_count.set(letter, (ran_count.get(letter) || 0) + 1);
    }
    for(let letter of ransomNote){
        if (!mag_count.has(letter) || ran_count.get(letter) > mag_count.get(letter)){
            return false}}
    return true
};

var canConstruct = function(ransomNote, magazine) {
    for (const char of magazine) {
        ransomNote = ransomNote.replace(char, "");
    }
    
    if (!ransomNote) return true;
    else return false;
};

var canConstruct = function(ransomNote, magazine) {
    const map = {};
    for(let letter of magazine) {
        if (!map[letter]) {
            map[letter] = 0;
        }
        map[letter]++;
    }
    
    for(let letter of ransomNote) {
        if(!map[letter]) {
            return false;
        } 
        map[letter]--;
    }
    return true;
};

var isIsomorphic = function(s, t) {
	const map1 = [];
	const map2 = [];
	for (let i = 0; i < s.length; i++) {
		if (map1[s[i]] != map2[t[i]])
			return false;

		map1[s[i]] = i + 1;
		map2[t[i]] = i + 1;
	}
	return true;
};

var isIsomorphic = function(s, t) {
    for (let i = 0; i < s.length; i++) {

        if (s.indexOf(s[i], i + 1) !== t.indexOf(t[i], i + 1)) {
            
            return false;
        }
    }
    return true;
};

var isIsomorphic = function(s, t) {
    let m = new Map()
    for (let i = 0; i < s.length; i++) {
        if (!m.has(s[i]))
            m.set(s[i], t[i])
        else {
            
            if (m.get(s[i]) != t[i]) {
                
                return false
            }
        }
    }
    return new Set([...m.values()]).size == m.size
};

var wordPattern = function(pattern, s) {
    words = s.split(' ')
    if(words.length != pattern.length){
        return false}
    let m = new Map()
    for (let i = 0; i < words.length; i++) {
        if (!m.has(words[i]))
            m.set(words[i], pattern[i])
        else {
            
            if (m.get(words[i]) != pattern[i]) {
                
                return false
            }
        }
    }
    return new Set(m.values()).size == m.size
};

var wordPattern = function(pattern, str) {
    const words = str.split(/\s+/);
    const map = new Map();
    
    if(words.length !== pattern.length) return false;
    if(new Set(words).size !== new Set(pattern).size) return false;
    
    for(let i = 0; i < pattern.length; i++) {
        if(map.has(pattern[i]) && 
           map.get(pattern[i]) !== words[i]) return false;
        map.set(pattern[i], words[i]);
    }
    return true;
};

var isAnagram = function(s, t) {
    let count = new Map()
    for (let letter of s) {
        count.set(letter, (count.get(letter) || 0) + 1);
    }
    for (let letter of t) {
        count.set(letter, (count.get(letter) || 0) - 1);
    }
    return [...count.values()].every(value => value === 0);
};

var isAnagram = function(s, t) {
    
        const counter = new Array(26).fill(0);
        
        for(let idx = 0; idx < s.length; idx++){
            counter[s.charCodeAt(idx)-97]++;
        }
        for(let idx = 0; idx < t.length; idx++){
            counter[t.charCodeAt(idx)-97]--;
        }
        
        
        
        
        for (let idx = 0; idx < 26; idx++) {
            if(counter[idx] != 0)
                return false;
        }
        return true;
};

var isAnagram = function(s, t) {
    if (t.length !== s.length) return false;
    const counts = {};
    for (let c of s) {
        counts[c] = (counts[c] || 0) + 1;
    }
    for (let c of t) {
        if (!counts[c]) return false;
        counts[c]--;
    }
    return true;
};

var isAnagram = function(s, t, m = {}) {
    for (let c of s) m[c] = (m[c] || 0) + 1;
    for (let c of t) if (!m[c]--) return false;
    return Object.values(m).every(v => !v);
};

var groupAnagrams = function(strs) {
    let map = new Map();

    for(let word of strs){
        let sorted_word = word.split('').sort().join('');
        if (!map.has(sorted_word)) {
            map.set(sorted_word, []);
        }
        map.get(sorted_word).push(word);
    }
    return [...map.values()]
};

var groupAnagrams = function(strs) {
    let obj = {};
    for (let str of strs) {
        let letters = str.split("").sort().join("");
        obj[letters] ? obj[letters].push(str) : obj[letters] = [str];
    }
    return Object.values(obj);
};

var groupAnagrams = function(strs) {
    let m = new Map();
    for (let str of strs) {
        let sorted = str.split("").sort().join("");
        if (m.has(sorted)) m.set(sorted, [...m.get(sorted), str]);
        else m.set(sorted, [str]);
    }
    return Array.from(m.values());
};



var groupAnagrams = function(strs) {
    let res = {};
    for (let str of strs) {
        let count = new Array(26).fill(0);
        for (let char of str) count[char.charCodeAt()-97]++;
        let key = count.join("#");
        res[key] ? res[key].push(str) : res[key] = [str];
    }
    return Object.values(res);
};



var twoSum = function(nums, target) {
    let numMap = new Map()
    for( let i = 0; i < nums.length; i++){
        let complement = target - nums[i]
        if(numMap.has(complement)){
            return [numMap.get(complement), i]}
        numMap.set(nums[i], i)}
};

var twoSum = function(nums, target) {
    let numMap = new Map()
    for( let i = 0; i < nums.length; i++){
        if(numMap.has(target - nums[i])){
            return [numMap.get(target - nums[i]), i]}
        numMap.set(nums[i], i)}
};

var twoSum = function(nums, target) {
    let hash = {};
    for( let i = 0; i < nums.length; i++){
        if(hash[target - nums[i]] !== undefined){
            return [hash[target - nums[i]], i]}
        hash[nums[i]] = i}
};


var isHappy = function(n) {
    let slow = squared(n)
	let fast = squared(squared(n))

	while(slow!==fast && fast!==1){
		slow = squared(slow)
		fast = squared(squared(fast))}

	return fast==1

};

var squared = function(n) {
	let result = 0
	while (n>0){
		let last = n % 10
		result += last * last
		n = Math.floor(n/10)}
	return result}

var isHappy = function(n) {
    var seen = {};
    while (n !== 1 && !seen[n]) {
        seen[n] = true;
        n = sumOfSquares(n);
    }
    return n === 1 ? true : false;
};

function sumOfSquares(numString) {
    return numString.toString().split('').reduce(function(sum, num) {
        return sum + Math.pow(num, 2);
    }, 0);
}

var isHappy = function(n) {
    if(n<10){
        if(n === 1 || n === 7){
            return true
        }
        return false
    }
    let total = 0
    while(n>0){
        let sq = n % 10
        total += sq**2
        n -= sq
        n /= 10
    }
    if(total === 1){
        return true
    }
    return isHappy(total)
};

var containsNearbyDuplicate = function(nums, k) {
    let s = new Set()
    for(let i = 0; i < nums.length; i++){
        if(!s.has(nums[i])){
            s.add(nums[i])}
        else{
            if(i - nums.indexOf(nums[i]) <= k){
                return true}
            nums[nums.indexOf(nums[i])] = -Infinity}}
    return false
};

var containsNearbyDuplicate = function(nums, k) {
    const hasmap = new Map();
    for (let idx = 0; idx < nums.length; idx++) {
        
        if (idx - hasmap.get(nums[idx]) <= k) {
            return true;
        }
        hasmap.set(nums[idx], idx);
    }
    return false;
};

var containsNearbyDuplicate = function(nums, k) {
	
	for (let i = 0; i < nums.length; i++) {
		for (let j = i + 1; j <= i + k && j < nums.length; j++) {
			if (nums[i] === nums[j]) 
				return true;
		}
	}

	return false;
};

var longestConsecutive = function(nums) {
    let longest = 0
    let num_set = new Set(nums)

    for(n of num_set){
        if(!num_set.has(n-1)){
            length = 1
            while(num_set.has(n+length)){
                length += 1}
            longest = Math.max(longest, length)}}
    
    return longest
};

var longestConsecutive = function(nums) {
    if (nums == null || nums.length === 0) return 0;
    
    const set = new Set(nums);
    let max = 0;
  
    for (let num of set) {
      if (set.has(num - 1)) continue;  
  
      let currNum = num;
      let currMax = 1;
  
      while (set.has(currNum + 1)) {
        currNum++;
        currMax++;
      }
      max = Math.max(max, currMax);
    }
  
    return max;
  }

  var summaryRanges = function(nums) {
    if(nums.length === 0){
        return []}
    let output =[]
    let start = nums[0]
    for(let i = 0; i < nums.length -1 ; i++){
        if(nums[i + 1] - 1 != nums[i]){
            if (start == nums[i]){
                output.push(start.toString())}
            else{
                output.push(`${start}->${nums[i]}`)}
            start = nums[i + 1]}}
    if(start == nums[nums.length -1]){
        output.push(start.toString())}
    else{
        output.push(`${start}->${nums[nums.length -1]}`)}
    return output
};

var summaryRanges = function(nums) {
    var t = 0
    var ans = []
    nums.push('#')
    for(var i=1;i<nums.length;i++)
        if(nums[i]-nums[t] !== i-t){
            if(i-t>1)
                ans.push(nums[t]+'->'+(nums[i-1]))
            else
                ans.push(nums[t].toString())
            t = i
        }
    return ans
}

var merge = function(intervals) {
    intervals.sort(function(a,b) {
        return a[0]-b[0]
    })
    let ans = []
    for (let interval of intervals){
        if (ans.length === 0  || ans[ans.length-1][1] < interval[0]){
            ans.push(interval)}
        else{
            ans[ans.length-1][1] = Math.max(ans[ans.length-1][1], interval[1])}}
    
    return ans

    
};

var merge = function(intervals) {
    if (!intervals.length) return intervals
    intervals.sort((a, b) => a[0] !== b[0] ? a[0] - b[0] : a[1] - b[1])
    var prev = intervals[0]
    var res = [prev]
    for (var curr of intervals) {
        if (curr[0] <= prev[1]) {
        prev[1] = Math.max(prev[1], curr[1])
        } else {
        res.push(curr)
        prev = curr
        }
    }
    return res
    
};


var insert = function(intervals, newInterval) {
    let output = []
    for (let interval of intervals){
        if (interval[1] < newInterval[0]){
            output.push(interval)}
        else if (newInterval[1] < interval[0]){
            output.push(newInterval)
            newInterval = interval}
        else{
            newInterval[0] = Math.min(newInterval[0],interval[0])
            newInterval[1] = Math.max(newInterval[1],interval[1])}}
    output.push(newInterval)
    return output
};

var insert = function(intervals, newInterval) {
    let [start, end] = newInterval;
    let left = [];
    let right = [];
    
    for (const interval of intervals) {
      const [first, last] = interval;
      
      
      if (last < start) left.push(interval);
      
      
      else if (first > end) right.push(interval);
      
      
      else {
        start = Math.min(start, first);
        end = Math.max(end, last);
      }
    }
    
    return [...left, [start, end], ...right]; 
  };

  var insert = function(intervals, newInterval) {
    const n = intervals.length
    
	
    let idx = 0
    
    while (idx < n) {
        if (intervals[idx][0] >= newInterval[0]) {
            break
        }
        idx++
    }
    
    intervals.splice(idx, 0, newInterval)
    
    
	
    let i = 0
    
	
    while (i < intervals.length - 1) {
       
	   
	   if (intervals[i][1] < intervals[i+1][0]) {
            i++
            continue
        }
        
		
        intervals[i][1] = Math.max(intervals[i+1][1], intervals[i][1])
        
		
		
        intervals.splice(i+1,1)
    }
    
    return intervals};

var findMinArrowShots = function(points) {
    points.sort(function(a,b) {
        return a[1]-b[1]
    })                                  
    let tally = 1
    let bow = points[0][1]                                       
    for (let [start, end] of points) {        
        if (bow < start){                 
            bow = end                     
            tally += 1  }  }                  
    return tally 
};

var isValid = function(s) {
    let chars = {"(":")","{":"}","[":"]"}
    let stack = []
    for (let char of s){
        if (char in chars){
            stack.push(chars[char])}
        else if (stack.length == 0 || stack.pop() != char){
            return false}}
    return stack.length == 0
};

var isValid = function(s) {
    let stack = []; 
    for (let c of s) { 
        if (c === '(' || c === '{' || c === '[') { 
            stack.push(c); 
        } else { 
            if (!stack.length || 
                (c === ')' && stack[stack.length - 1] !== '(') || 
                (c === '}' && stack[stack.length - 1] !== '{') ||
                (c === ']' && stack[stack.length - 1] !== '[')) {
                return false; 
            }
            stack.pop(); 
        }
    }
    return !stack.length; 
                          
};

var simplifyPath = function(path) {
    let dirOrFiles = []
    path = path.split("/")
    for(let elem of path){
        if (dirOrFiles && elem == ".."){
            dirOrFiles.pop()}
        else if (![ ".", "", ".."].includes(elem)){
            dirOrFiles.push(elem)}}
            
    return "/" + dirOrFiles.join('/')

};

var simplifyPath = function(path) {
    const stack = [];
    const directories = path.split("/");
    for (const dir of directories) {
        if (dir === "." || !dir) {
            continue;
        } else if (dir === "..") {
            if (stack.length > 0) {
                stack.pop();
            }
        } else {
            stack.push(dir);
        }
    }
    return "/" + stack.join("/");
    
};
var MinStack = function() {
    this.stack = []
    this.currentMin = Infinity
    this.prevMins = [] 
};

MinStack.prototype.push = function(val) {
    this.stack.push(val)
    if (val <= this.currentMin){
        this.prevMins.push(this.currentMin)
        this.currentMin = val}
};

MinStack.prototype.pop = function() {
    if (this.stack[this.stack.length-1] == this.currentMin){
        this.currentMin = this.prevMins.pop()}
    this.stack.pop()
};

MinStack.prototype.top = function() {
    return this.stack[this.stack.length-1]
};

MinStack.prototype.getMin = function() {
    return this.currentMin
};

var MinStack = function() {
    this.elements = [];
  };
  
    MinStack.prototype.push = function(x) {
    this.elements.push({
      value: x,
      min: this.elements.length === 0 ? x : Math.min(x, this.getMin()),
    });
  };
    MinStack.prototype.pop = function() {
    this.elements.pop();
  };
    MinStack.prototype.top = function() {
    return this.elements[this.elements.length - 1].value;
  };
    MinStack.prototype.getMin = function() {
    return this.elements[this.elements.length - 1].min;
  };

  class LinkedNode {
    constructor(val, min, next = null) {
        this.val = val
        this.min = min
        this.next = next
    }
}

class MinStack {
    constructor() {
        this.head = null
    }
    push(val) {
        if (!this.head) this.head = new LinkedNode(val, val)
        else this.head = new LinkedNode(val, Math.min(val, this.head.min), this.head)
    }
    pop() {
        this.head = this.head.next
    }
    top() {
        return this.head.val
    }
    getMin() {
        return this.head.min
    }
}

var evalRPN = function(tokens) {
    let op = {'+': (x, y) => y + x, 
            '-': (x, y) => y - x,
            '*': (x, y) => y * x,
            '/': (x, y) => y / x < 0 ? Math.ceil(y / x): Math.floor(y / x)}
    let stack = []
    for (let t of tokens){
        if(t in op){
            t = op[t](stack.pop(), stack.pop())}
        stack.push(Number(t))}
    return stack[0]
};

var evalRPN = function(tokens) {
    let arr =[]
    for(let i =0; i<tokens.length;i++){
        if(tokens[i] == "+"){
            let sec = +arr.pop()
            let fir = +arr.pop()
            arr.push(sec + fir)
        }
        else if(tokens[i] == "-"){
            let sec = +arr.pop()
            let fir = +arr.pop()
            arr.push(fir - sec)
        }
        else if(tokens[i] == "*"){
            let sec = +arr.pop()
            let fir = +arr.pop()
            arr.push(sec * fir)
        }
        else if(tokens[i] == "/"){
            let sec = +arr.pop()
            let fir = +arr.pop()
            arr.push(Math.trunc(fir/sec))
        }
        else arr.push(tokens[i])
    }
    return arr[0]
};

var calculate = function(s) {
    let output = 0
    let curr = 0
    let sign = 1
    let stack = []
    for( let char of s){
        if (!isNaN(char) && char !==' ') {
            curr = curr*10 + Number(char)}
        else if ( char == '+' || char == '-'){
            output += curr * sign
            curr = 0
            if (char == '+'){
                sign = 1}
            else{
                sign = -1}}
        else if ( char =='('){
            stack.push(output)
            stack.push(sign)
            sign = 1
            output = 0}
        else if (char == ')'){
            output += curr * sign
            output = stack.pop()*output + stack.pop()
            curr = 0}}
    return output + curr * sign
    
};

var calculate = function(s) {
    let sign = 1, sum = 0;
    
    const stack = []; 
    for (let i = 0; i < s.length; i += 1) {
        if (s[i] >= '0' && s[i] <= '9') {
            let num = 0
			
            while (s[i] >= '0' && s[i] <= '9') {
                num = (num * 10) + (s[i] - '0');
                i += 1;
            }
			
            sum += (num * sign);
			
            i -= 1;
        } else if (s[i] === '+') {
            sign = 1;
        } else if (s[i] === '-') {
            sign = -1;
        } else if (s[i] === '(') {
			
			
            stack.push(sum);
            stack.push(sign);
            sum = 0
			
            sign = 1;
        } else if (s[i] === ')') {
			
			
            sum = stack.pop() * sum;
            sum += stack.pop();
        }
    }
    
    return sum;
};

var calculate = function(s) {
    let res = 0, sum = 0, sign = 1;
    let myStack = [];
    myStack.push(1);
    const isDigit = (ch) => {
        return ch >= '0' && ch <= '9';
    }
    for(let ch of s){
        if(isDigit(ch)) sum = sum * 10 + (ch - '0');
        else{
            res += sum * sign * myStack[myStack.length - 1];
            sum = 0;
            if(ch === '-') sign = -1;
            else if(ch === '+') sign = 1;
            else if(ch === '(') {myStack.push(myStack[myStack.length - 1] * sign); sign = 1;}
            else if(ch === ')') myStack.pop(); 
        }
    }
    return res += (sign * sum);
};

function ListNode(val) {
    this.val = val;
    this.next = null;
    }

var hasCycle = function(head) {

    if (!head){
        return false}
    let slow = head
    let fast = head.next

    while (slow!==fast && fast){
        slow = slow.next
        fast = fast.next
        if (fast){
            fast = fast.next}}
    return !!fast
    
};

var hasCycle = function(head) {

    let fast = head;
    while (fast && fast.next) {
      head = head.next;
      fast = fast.next.next;
      if (head === fast) return true;
    }
    return false;
  };

function ListNode(val, next) {
    this.val = (val===undefined ? 0 : val)
    this.next = (next===undefined ? null : next)
}

  var addTwoNumbers = function(l1, l2) {
    let res = new ListNode(), dummy = res
    let carry = 0
    while (l1|| l2){
        let v1 = 0, v2 = 0
        if (l1){
            v1 = l1.val;
            l1 = l1.next}
        if (l2){
            v2 = l2.val;
            l2 = l2.next}
        
        let val = carry + v1 + v2
        res.next = new ListNode(val%10)
        res = res.next
        carry = Math.floor(val/10)
    }
    if (carry){
        res.next = new ListNode(carry)}
        
    return dummy.next
};

var addTwoNumbers = function(l1, l2) {
    const iter = (n1, n2, rest = 0) => {
        if (!n1 && !n2 && !rest) return null;
        const newVal = (n1?.val || 0) + (n2?.val || 0) + rest;
        const nextNode = iter(n1?.next, n2?.next, Math.floor(newVal / 10));
        return new ListNode(newVal % 10, nextNode);
    }
    return iter(l1, l2);
};


var addTwoNumbers = function(l1, l2, carry = 0) {
    if (!l1 && !l2 && !carry) return null;
    var newVal = (l1?.val || 0) + (l2?.val || 0) + carry;
    carry = Math.floor(newVal/10)
    return new ListNode(newVal % 10, addTwoNumbers(l1?.next, l2?.next, carry))
};

var mergeTwoLists = function(l1, l2) {
    if (!l1 || !l2){
        return l1 || l2}
    
    if (l1.val <= l2.val){
        l1.next = mergeTwoLists(l1.next, l2)
        return l1}
    else{
        l2.next = mergeTwoLists(l1, l2.next)
        return l2}
};

var mergeTwoLists = function(list1, list2) {
    let dummy = new ListNode()
    let cur = dummy
    while (list1 && list2){               
        if (list1.val < list2.val){
            cur.next = list1
            list1 = list1.next
            cur = cur.next}
        else{
            cur.next = list2
            list2 = list2.next
            cur = cur.next}}
            
    if (list1 || list2){
        if (list1){
             cur.next = list1}
        else{cur.next=list2}}
        
    return dummy.next}

var copyRandomList = function(head) {
    if(!head) {
        return null;
    }
    const clones = new Map();
    let n = head;
    while(n) {
        clones.set(n, new Node(n.val));
        n = n.next
    }
    n = head;
    while(n) {
        clones.get(n).next = clones.get(n.next) || null;
        clones.get(n).random = clones.get(n.random) || null;
        n = n.next
    }
    return clones.get(head);
};

var copyRandomList = function(head) {
    let visited = new Map();
    
    let helper = (node) => {
        if (!node) return null;
        if (visited.has(node)) return visited.get(node);
        
        let newNode = new Node(node.val);
        visited.set(node, newNode);
        newNode.next = helper(node.next);
        newNode.random = helper(node.random);
        return newNode;
    }
    return helper(head);
};


var copyRandomList = function(head) {
    if (!head){
        return null}
    let pointer = head;
    while (pointer) {
        const originalNext = pointer.next;
        pointer.next = new Node(pointer.val, originalNext);
        pointer = pointer.next.next;
    }
    pointer = head
    while (pointer) {
        const copy = pointer.next; 
        copy.random = pointer.random ? pointer.random.next : null;
        pointer = pointer.next.next;
    }

    pointer = head;
    const dummy = new Node(null, head.next);
    let copyPointer;
    while (pointer) {
        
        copyPointer = pointer.next;
        pointer.next = pointer.next.next;
        copyPointer.next = copyPointer.next ? copyPointer.next.next : null;
        
        pointer = pointer.next;
}
    return dummy.next

};

var reverseBetween = function(head, left, right) {
    let dummy = new ListNode(0)
    dummy.next = head
    
    let pre = dummy
    let cur = dummy.next
    
    for (let i =1; i <left; i++){
        cur = cur.next
        pre = pre.next}
    
    for (let i =0; i <right-left; i++){
        let temp = cur.next
        cur.next = temp.next
        temp.next  = pre.next
        pre.next = temp}
    
    return dummy.next
};

var reverseBetween = function(head, m, n) {
    let start = head, cur = head;
    let i = 1;
    while (i < m) {
        start = cur;
        cur = cur.next;
        i++;
    }
    let prev = null, tail = cur;
    while (i <= n) {
        let next = cur.next;
        cur.next = prev;
        prev = cur;
        cur = next;
        i++;
    }
    start.next = prev;
    tail.next = cur;
    return m == 1 ? prev : head; 
};

var reverseKGroup = function(head, k) {
    var dummy = new ListNode(0);
    dummy.next = head;
    var prevGroupTail = dummy;

    while (head) {
        var groupStart = head;
        var groupEnd = getGroupEnd(head, k);

        if (!groupEnd)
            break;

        prevGroupTail.next = reverseList(groupStart, groupEnd.next);
        prevGroupTail = groupStart;
        head = prevGroupTail.next;
    }
var newHead = dummy.next;
return newHead;
}

var getGroupEnd = function(head, k) {
    while (head && k > 1) {
        head = head.next;
        k--;
    }
    return head;
}

var reverseList = function(head, stop) {
    var prev = stop;
    while (head !== stop) {
        var next = head.next;
        head.next = prev;
        prev = head;
        head = next;
    }
    return prev;
}

function reverseKGroup(head, k) {
    if (!head) return null;
    var tail = head;
    for (var i = 1; i < k; i++) {
      tail = tail.next;
      if (!tail) return head;
    }
    var next = tail.next;
    tail.next = null;
    reverse(head);
    head.next = reverseKGroup(next, k);
    return tail;
  }
  
function reverse(curr) {
    var prev = null;
    while (curr) {
    var next = curr.next;
    curr.next = prev;
    prev = curr;
    curr = next;
    }
    return prev;
}

var removeNthFromEnd = function(head, n) {
    let dummy = new ListNode(0,head)
    let traverse = head
    let counter = 1
    while (traverse){
        traverse = traverse.next
        counter +=1}
    traverse = dummy
    while (counter > n+1){
        traverse = traverse.next
        counter -=1  }
    traverse.next = traverse.next.next
    return dummy.next

};

var removeNthFromEnd = function(head, n) {
    let dummy = new ListNode(0,head)
    let Second_pointer = dummy
    for(let i = 0; i < n-1; i++){
        head = head.next}
    while (head.next){
        head = head.next
        Second_pointer = Second_pointer.next}
    Second_pointer.next =  Second_pointer.next.next
    return dummy.next


};

var deleteDuplicates = function(head) {
    let dummy = new ListNode(0,head)
    let Second_pointer = dummy
    while (head && head.next){
        if (head.val == head.next.val){
            while (head.next && head.val == head.next.val){
                head = head.next}
            Second_pointer.next = head.next
            head = head.next
            continue}
        head = head.next
        Second_pointer = Second_pointer.next }
    return dummy.next
};

var deleteDuplicates = function(head) {
    
    if (head == null || head.next == null)
        return head;
    
    var fake = new ListNode(0);
    fake.next = head;
    var curr = fake;
    
    while(curr.next != null && curr.next.next != null){         
        
        
        if(curr.next.val == curr.next.next.val) {
            let duplicate = curr.next.val;
            
            while(curr.next !=null && curr.next.val == duplicate) {
                
                curr.next = curr.next.next;
            }
        }
        
        else{
            curr = curr.next;
        }
    }
    return fake.next;       
};

var rotateRight = function(head, k) {
    if (!head){
    return head}
    let cur= head
    let length =1
    while (cur.next){
        cur = cur.next
        length+=1 }
    cur.next = head
    k= length - (k%length)
    while (k>0){
        cur=cur.next
        k-=1}
    let newhead = cur.next
    cur.next= null
    return newhead

};

var partition = function(head, x) {
    let dummy_small = new  ListNode(0)
    let small_node = dummy_small
    let dummy_big = new ListNode(0)
    let big_node = dummy_big
    while (head){
        if (head.val < x){
            small_node.next = head
            small_node = head}
        else{
            big_node.next = head
            big_node = head}
        head = head.next}
    
    big_node.next = null
    small_node.next = dummy_big.next
    return dummy_small.next
    
};

class LRUCache {
    constructor(capacity) {
      this.cache = new Map();
      this.capacity = capacity;
    }
  
    get(key) {
      if (!this.cache.has(key)) return -1;
  
      const v = this.cache.get(key);
      this.cache.delete(key);
      this.cache.set(key, v);
      return this.cache.get(key);
    };
  
    put(key, value) {
      if (this.cache.has(key)) {
        this.cache.delete(key);
      }
      this.cache.set(key, value);
      if (this.cache.size > this.capacity) {
        this.cache.delete(this.cache.keys().next().value);  
      }
    };
  }

  class ListNode {
    constructor(key, value) {
        this.key = key;
        this.value = value;
        this.prev = null;
        this.next = null;
    }
}

class LRUCache {
    constructor(capacity) {
        this.dic = new Map(); 
        this.capacity = capacity;
        this.head = new ListNode(0, 0);
        this.tail = new ListNode(-1, -1);
        this.head.next = this.tail;
        this.tail.prev = this.head;
    }

    get(key) {
        if (this.dic.has(key)) {
            const node = this.dic.get(key);
            this.removeFromList(node);
            this.insertIntoHead(node);
            return node.value;
        } else {
            return -1;
        }
    }

    put(key, value) {
        if (this.dic.has(key)) { 
            const node = this.dic.get(key);
            this.removeFromList(node);
            this.insertIntoHead(node);
            node.value = value; 
        } else {
            if (this.dic.size >= this.capacity) {
                this.removeFromTail();
            }
            const node = new ListNode(key, value);
            this.dic.set(key, node);
            this.insertIntoHead(node);
        }
    }

    removeFromList(node) {
        node.prev.next = node.next;
        node.next.prev = node.prev;
    }

    insertIntoHead(node) {
        const headNext = this.head.next;
        this.head.next = node;
        node.prev = this.head;
        node.next = headNext;
        headNext.prev = node;
    }

    removeFromTail() {
        if (this.dic.size === 0) return;
        const tail_node = this.tail.prev;
        this.dic.delete(tail_node.key);
        this.removeFromList(tail_node);
    }
}

var maxDepth = function(root) {
    if(!root){
        return 0}
    let check = [root]
    let output = 0
    while (check.length != 0){
        let nodes = check
        check = []
        for (let node of nodes){
            if (node.left){
                check.push(node.left)}
            if (node.right){
                check.push(node.right)}}
        output += 1}
    return output
    
};

var maxDepth = function(root) {
    if (!root){
        return 0}
    return Math.max(maxDepth(root.left), maxDepth(root.right)) + 1

    
};

var maxDepth = function(root) {
    if (!root) return 0;
    const queue = [root];
    let depth = 0;
    while (queue.length !== 0) {
        depth++;
        const len = queue.length;
        for (let i = 0; i < len; i++) {
            if (queue[i].left) queue.push(queue[i].left);
            if (queue[i].right) queue.push(queue[i].right);
        }
        queue.splice(0, len);
    }
    return depth;
};

var maxDepth = function(root) {
    if(!root) return 0;
    
    
    let levels = 0, queue = [];
    queue.push(root);
    
    while(queue.length > 0){
        let count = queue.length;
        
        for(let i = 0; i < count; i++){
            const node = queue.shift();
            if(node.right) queue.push(node.right);
            if(node.left) queue.push(node.left);
            
        }
        levels++;
    }
    return levels;
};

var isSameTree = function(p, q) {
    let stack = [[p, q]]
    while(stack.length > 0 ){
        let [first, second] = stack.pop();
        if (!first && !second){
            continue
        }
        else if (!first || !second){
            return false}
        else{
            if (first.val != second.val){
                return false}
            stack.push([first.left,second.left])
            stack.push([first.right, second.right])}}
    return true
    
};

var isSameTree = function(p, q) {
    if (p && q){
        return p.val == q.val && isSameTree(p.left,q.left) && isSameTree(p.right, q.right)
    }
    else if (!p && !q) {
        return true;
    } else {
        return false;}

};

var isSameTree = function(p, q) {
    const stack1 = [], stack2 = [];
    while (p || q || stack1.length || stack2.length) {
        while (p) {
            stack1.push(p);
            p = p.left
        }
        while (q) {
            stack2.push(q);
            q = q.left;
        }
        p = stack1.pop();
        q = stack2.pop();
        if (!p && !q) {
            continue;
        }
        if (!p || !q || p.val !== q.val) {
            return false;
        }
        stack1.push(null);
        stack2.push(null);
        p = p.right;
        q = q.right;
    }
    return true;
};


var invertTree = function(root) {
    
    if(root == null){
        return root
    }
    
    invertTree(root.left)
    
    invertTree(root.right)
    
    const curr = root.left
    root.left = root.right
    root.right = curr
    return root         
};

var invertTree = function(root) {
    let layer = [root]
    while (layer.length > 0){
        let node = layer.pop()
        if(node){
            let cur = node.left 
            node.left = node.right
            node.right = cur
            layer.push(node.left)
            layer.push(node.right)}}
    
    return root
    
};

function invertTree(root) {
    const stack = [root];
  
    while (stack.length) {
      const n = stack.pop();
      if (n != null) {
        [n.left, n.right] = [n.right, n.left];
        stack.push(n.left, n.right);
      }
    }
  
    return root;
  }

var isSymmetric = function(root) {
    if (!root){
        return true}

    return isSame(root.left, root.right)

    function isSame(leftroot, rightroot){
        if (leftroot === null && rightroot === null){
            return true}
        if (leftroot === null || rightroot === null){
            return false}
        if (leftroot.val !== rightroot.val){
            return false}
        return isSame(leftroot.left, rightroot.right) && isSame(leftroot.right, rightroot.left)}
};

function TreeNode(val, left, right) {
    this.val = (val===undefined ? 0 : val)
    this.left = (left===undefined ? null : left)
    this.right = (right===undefined ? null : right)
}

var isSymmetric = function(root) {
    let stack = []
    if (root){
    stack.push([root.left, root.right])
    }
    while(stack.length > 0){
        let [left, right] = stack.pop()
        
        if (left && right){
            if (left.val !== right.val) return false
            stack.push([left.left, right.right])
            stack.push([right.left, left.right])}
    
        else if (left || right)return false}
    
    return true
};

var buildTree = function(preorder, inorder) {
    let idx_map = {}
    inorder.forEach((value,index) => idx_map[value] = index)
    preorder.reverse()
    return helper(0, preorder.length - 1, preorder, idx_map)

    function helper(left, right, preorder,  idx_map){
        if (left > right) return null
        let root_val = preorder.pop()
        let root = new TreeNode(root_val)
        root.left = helper(left,idx_map[root_val]-1, preorder, idx_map)
        root.right = helper(idx_map[root_val]+1, right, preorder, idx_map)
        return root
    }
    
};

var buildTree = function(preorder, inorder) {
    p = i = 0
    build = function(stop) {
        if (inorder[i] != stop) {
            var root = new TreeNode(preorder[p++])
            root.left = build(root.val)
            i++
            root.right = build(stop)
            return root
        }
        return null
    }
    return build()
};

var buildTree = function(inorder, postorder) {
    let idx_map = {}
    inorder.forEach((value,index) => idx_map[value] = index)
    return helper(0, postorder.length - 1, postorder, idx_map)

    function helper(left, right, postorder,  idx_map){
        if (left > right) return null
        let root_val = postorder.pop()
        let root = new TreeNode(root_val)
        root.right = helper(idx_map[root_val]+1, right, postorder, idx_map)
        root.left = helper(left,idx_map[root_val]-1, postorder, idx_map)
        return root
    }
    
};

var buildTree = function(inorder, postorder) {
    build = function(stop) {
        if (inorder[inorder.length-1] != stop) {
            var root = new TreeNode(postorder.pop())
            root.right = build(root.val)
            inorder.pop()
            root.left = build(stop)
            return root
        }
        return null
    }
    return build()
};


var connect = function(root) {
    let layer = [root,null]
    if (!root){
        return null}
    while (layer.length > 0 ){
        let node = layer.shift()
        if (!node){
            if (layer.length >0){
                layer.push(null)}
            continue}
        node.next = layer[0]
        if (node.left){
            layer.push(node.left)}
        if (node.right){
            layer.push(node.right)}}

    return root
    
};


var connect = function(root) {
    if (!root){
        return null}
        
    let curr=root
    let dummy= new Node(-999)        
    let head=root        

    while (head){
        curr=head
        let prev=dummy 
        while (curr){
            if (curr.left){
                prev.next=curr.left
                prev=prev.next}
            if (curr.right){
                prev.next=curr.right
                prev=prev.next}                                              
            curr=curr.next}
        head=dummy.next 
        dummy.next=null}
    return root}


var connect = function(root) {
    let curr = root;
    
    while (curr != null) {
        let start = null; 
        let prev = null;
    
        while (curr != null) { 
            if (start == null) { 
                if (curr.left) start = curr.left;
                else if (curr.right) start = curr.right;
                
                prev = start; 
            }
            
            if (prev != null) {
                if (curr.left && prev != curr.left) {
                    prev = prev.next = curr.left; 
                }
                if (curr.right && prev != curr.right) {
                    prev = prev.next = curr.right;
                }
            }

            curr = curr.next; 
        }
        
        curr = start; 
    }
    
    return root;
};


var flatten = function(root) {
    let cur = root
    while (cur){
            if (cur.left){
                    let prev = cur.left
                    while (prev.right){
                            prev = prev.right}
                    
                    prev.right = cur.right
                    cur.right = cur.left
                    cur.left = null}
            
            cur = cur.right
    }
};


var flatten = function(root) {
    let previous_right = null
    function helper(root){
            if (root){
                    helper(root.right)
                    helper(root.left)
                    root.right = previous_right
                    previous_right = root
                    root.left = null}}
    helper(root)}


var flatten = function(root) {
    let head = null, curr = root
    while (head != root) {
        if (curr.right === head) curr.right = null
        if (curr.left === head) curr.left = null
        if (curr.right) curr = curr.right
        else if (curr.left) curr = curr.left
        else curr.right = head, head = curr, curr = root
    }
};


var hasPathSum = function(root, targetSum) {
    let layer = [root]
    while (layer.length !== 0 && root){
        node = layer.pop()
        if (!node.left && !node.right && node.val == targetSum){
            return true}
        if (node.left){
            node.left.val += node.val
            layer.push(node.left)}
        if (node.right){
            node.right.val += node.val
            layer.push(node.right)}}
    return false

    
};


var hasPathSum = function(root, sum) {
	if (!root){
		return false}
	if (!root.left && !root.right && root.val == sum){
		return true}
	return hasPathSum(root.left, sum - root.val) || hasPathSum(root.right, sum - root.val)

};

var sumNumbers = function(root) {
    let layer = [[root,root.val]]
    let Total = 0
    while (layer.length !== 0 && root){
        [node, node_value] = layer.pop()
        if (!node.left && !node.right){
            Total += node_value}
        if (node.left){
            left_value = node_value*10 + node.left.val
            layer.push([node.left,left_value])}
        if (node.right){
            right_value = node_value*10 + node.right.val
            layer.push([node.right,right_value])}}
    return Total

    
};


var sumNumbers = function(root) {
    let total = 0
    function helper(node,total){
        if (! node){
            return 0}
        total = total*10 + node.val
        if (!node.left && !node.right){
            return total}
        return helper(node.left,total) + helper(node.right,total)}
    return helper(root,total)

};


var sumNumbers = function(root) {
    let tot_sum = 0
    let cur = 0
    let depth = 0
    while (root){
        if (root.left){
            let pre = root.left
            depth = 1
            while (pre.right && pre.right !== root){
                pre = pre.right
                depth += 1}
            if (!pre.right){
                pre.right = root
                cur = cur * 10 + root.val
                root = root.left}
            else{
                pre.right = null
                if (!pre.left) tot_sum += cur
                cur = Math.floor(cur / Math.pow(10, depth));
                root = root.right}}
        else{
            cur = cur * 10 + root.val
            if (!root.right) tot_sum += cur
            root = root.right}}
    return tot_sum

};

var sumNumbers = function(root) {
    function traverse(node, num) {
        if(!node) return null;
        num += node.val
        if(!node.left && !node.right) return +num;
        return traverse(node.left, num) + traverse(node.right, num);
    }
    return traverse(root, '');
};

var maxPathSum = function(root) {
    let res = -Infinity
    helper(root)
    return res 
        
    function helper(root){
        if(!root){
            return 0}
        let left = helper(root.left)
        let right = helper(root.right)
        res = Math.max(res, root.val + left + right)
        return Math.max(root.val + Math.max(left, right), 0)}
    
};

var BSTIterator = function(root) {
    this.root = root
    
};

BSTIterator.prototype.next = function() {
    while (this.root){
        if (this.root.left){
            let pre = this.root.left
            while (pre.right && pre.right !== this.root){
                pre = pre.right}
            if (!pre.right){
                pre.right = this.root
                this.root = this.root.left}
            else{
                to_return = this.root.val
                pre.right = null
                this.root = this.root.right
                return to_return}}
        else{
            to_return = this.root.val
            this.root = this.root.right
            return to_return}}
};

BSTIterator.prototype.hasNext = function() {
    if (this.root){
        return true}
    return false
};

var BSTIterator = function(root) {
    this.stack = []
    while (root){
        this.stack.push(root)
        root = root.left}

};

BSTIterator.prototype.next = function() {
    let node = this.stack.pop()
    if (node.right){
        let current = node.right
        while (current){
            this.stack.push(current)
            current = current.left}}
    return node.val

};

BSTIterator.prototype.hasNext = function() {
return this.stack.length > 0

};

function* inorder(node) {
    if (node) {
      yield* inorder(node.left);
      yield node.val;
      yield* inorder(node.right);
    }
  }
  
  class BSTIterator {
      constructor(root) {
          this.iter = inorder(root);
          this.nxt = this.iter.next().value;
      }
  
      next() {
          const res = this.nxt;
          this.nxt = this.iter.next().value;
          return res;
      }
  
      hasNext() {
          return this.nxt !== undefined;
      }
  }

var countNodes = function(root) {
    if (!root){
    return 0}

    let l = root
    let r = root
    let heightL = 0
    let heightR = 0

    while (l){
    heightL += 1
    l = l.left}

    while (r){
    heightR += 1
    r = r.right}

    if (heightL == heightR){
    return Math.pow(2, heightL) - 1}
    return 1 + countNodes(root.left) + countNodes(root.right)
    
};

var lowestCommonAncestor = function(root, p, q) {
    if (!root || root == p || root == q)
        return root

    let l = lowestCommonAncestor(root.left, p, q)
    let r = lowestCommonAncestor(root.right, p, q)

    if (l && r){
        return root}
    return l || r
    
};


var lowestCommonAncestor = function(root, p, q) {
    var pNode;
    var qNode;
    
    
    
    
    
    
    root.level = 0;
    let stack = [root];

    
    while (stack.length && !(pNode && qNode)) {
        
        let node = stack.pop();
        
        if (node) {
            
            
            if (node.val === p.val) {
                pNode = node;
            }
            
            
            if (node.val === q.val) {
                qNode = node;
            }
    
            
            if (node.right) {
                node.right.level = node.level + 1;
                node.right.parent = node;
                stack.push(node.right);
            }
            if (node.left) {
                node.left.level = node.level + 1;
                node.left.parent = node;
                stack.push(node.left);
            }
        }
    }
    
    
    
    
    
    while (pNode.val !== qNode.val) {
        
        if (pNode.level > qNode.level) {
            pNode = pNode.parent;
        } else if (pNode.level < qNode.level) {
            qNode = qNode.parent;
        } else {
            qNode = qNode.parent;
            pNode = pNode.parent;
        }
    }
    
    return pNode;
}



var rightSideView = function(root) {
    function solve(root, lvl){
        if (root){
            if (res.length ==lvl){
                res.push(root.val)}
            solve(root.right, lvl + 1)
            solve(root.left, lvl + 1)}
        return }

    let res = []
    solve(root,0)
    return res
};


var rightSideView = function(root) {
    if (!root){
        return []}
    let result = []
    let queue = [root]
    while (queue.length > 0){
        let level_len = queue.length
        for(let i = 0; i < level_len; i++){
            node = queue.shift()
            if (i == level_len - 1){
                result.push(node.val)}
            if (node.left){
                queue.push(node.left)}
            if (node.right){
                queue.push(node.right)}}}
    return result
};

var rightSideView = function(root) {
    if(!root) return []
    
    let queue = [root];
    const result = [root.val]
    
    while(queue.length) {
        const next = [];
        
        for(let node of queue) {
            if(node.left) next.push(node.left);
            if(node.right) next.push(node.right);
        }
        if(next.length) result.push(next[next.length-1].val);
        queue = next;
    }
    return result;

};

var averageOfLevels = function(root) {
    let lvlcnt = {};
    let lvlsum = {};

    function dfs(node = root, level=0) {
        if (!node) return;

        lvlcnt[level] = (lvlcnt[level] || 0) + 1;
        lvlsum[level] = (lvlsum[level] || 0) + node.val;

        dfs(node.left, level + 1);
        dfs(node.right, level + 1);
    }

    dfs();

    
    const levels = Object.keys(lvlcnt);

    return levels.map(level => lvlsum[level] / lvlcnt[level]);
};

var averageOfLevels = function(root) {
    
    let queue = [root];
    const result = []
    
    while(queue.length) {
        const next = [];
        let sum = 0
        
        for(let node of queue) {
            if(node.left) next.push(node.left);
            if(node.right) next.push(node.right);
            sum += node.val
        }
        result.push(sum/queue.length)
        queue = next;
    }
    return result;

};

var averageOfLevels = function(root) {
    let q = [root], ans = []
    while (q.length) {
        let qlen = q.length, row = 0
        for (let i = 0; i < qlen; i++) {
            let curr = q.shift()
            row += curr.val
            if (curr.left) q.push(curr.left)
            if (curr.right) q.push(curr.right)
        }
        ans.push(row/qlen)
    }
    return ans
};

var averageOfLevels = function(root) {
    const sum = [];
    const count = []
    const traverse = (node, i) => {
        if(sum[i] === undefined) sum[i] = 0;
        if(count[i] === undefined) count[i] = 0;
        sum[i] += node.val;
        count[i]++;
        if(node.left) traverse(node.left, i + 1);
        if(node.right) traverse(node.right, i + 1)
    }
    traverse(root, 0)
    for(let i = 0; i < sum.length; i++){
        sum[i] = sum[i] / count[i]
    }
    return sum;
};

var levelOrder = function(root) {    
    if (!root){
        return []  } 
    let queue = [root];
    const result = []
    
    while(queue.length) {
        const next = [];
        let row = []
        
        for(let node of queue) {
            if(node.left) next.push(node.left);
            if(node.right) next.push(node.right);
            row.push([node.val])
        }
        result.push(row)
        queue = next;
    }
    return result;

};

var levelOrder = function(root) {    
    let q = [root], ans = []
    while (q[0]) {
        let qlen = q.length, row = []
        for (let i = 0; i < qlen; i++) {
            let curr = q.shift()
            row.push(curr.val)
            if (curr.left) q.push(curr.left)
            if (curr.right) q.push(curr.right)
        }
        ans.push(row)            
    }
    return ans
};


var zigzagLevelOrder = (root) => {
    let res = [];
  
    const go = (node, lvl) => {
      if (node == null) return;
      if (res[lvl] == null) res[lvl] = [];
  
      if (lvl % 2 === 0) {
        res[lvl].push(node.val);
      } else {
        res[lvl].unshift(node.val);
      }
  
      go(node.left, lvl + 1);
      go(node.right, lvl + 1);
    };
  
    go(root, 0);
    return res;
  };


var zigzagLevelOrder = function(root) {
    if(!root) return [];
    let queue = [root];
    let output = [];
    let deep = 0;
    while(queue.length > 0){
        const size = queue.length;
        const level = [];
      
        for(let i=0; i< size; i++){
            const node = queue.shift();
            if(deep % 2 == 0) level.push(node.val);
            else level.unshift(node.val);
        
            if(node.left) queue.push(node.left)
            if(node.right) queue.push(node.right)
      }
    output.push(level)
    deep++;
    }
    
    
    return output
    
    
  };

var zigzagLevelOrder = function(root) {
    if (!root) return [];
    
    let result = [], q = [root], level = 0;
    
    while (q.length) {
        let size = q.length, currLevel = [];
        for (let i = 0; i < size; i++) {
            let node = q.shift();
            currLevel.push(node.val);
            if (node.left) q.push(node.left);
            if (node.right) q.push(node.right);
        }
        if (level % 2 === 1) currLevel.reverse();
        result.push(currLevel);
        level++;
    }
    return result;
};

var getMinimumDifference = function(root) {
    function* inorder(node) {
        if (node) {
            yield* inorder(node.left);
            yield node.val;
            yield* inorder(node.right);
        }
  }
    let minimum = Infinity
    let generator = inorder(root)
    let current = generator.next()
    let next_node = generator.next()
    while (!next_node.done){
        minimum = Math.min(minimum,Math.abs(current.value-next_node.value))
        current = next_node
        next_node = generator.next()}
    return minimum
  
};


var getMinimumDifference = function(root) {
    let cur = root, stack = [], minDiff = Infinity, prev = -Infinity
    
    while (stack.length !== 0  || cur){
        while (cur){
            stack.push(cur)
            cur = cur.left}
        let node = stack.pop()
        minDiff = Math.min(minDiff, node.val - prev)
        prev = node.val
        cur = node.right}
    
    return minDiff}

const getMinimumDifference = root => recurse( -Infinity, Infinity, root )

const recurse = ( less, more, node ) =>
    ! node
        ? Infinity
        : Math.min(
            node.val - less,
            more - node.val,
            recurse( less, node.val, node.left ),
            recurse( node.val, more, node.right ),
        )


var kthSmallest = function(root, k) {
    function* inorder(node) {
        if (node) {
            yield* inorder(node.left);
            yield node.val;
            yield* inorder(node.right);
        }
    }
    let generator = inorder(root)
    for(let i = 0; i < k; i++){
        output = generator.next()}
    return output.value
    
};

var kthSmallest = function(root, k) {
    let vals = [];
    (function dfs(node) {
      if (vals.length !=k) { 
        if(node.left) dfs(node.left); 
        vals.push(node.val); 
        if (node.right) dfs(node.right); 
      }  
    })(root) 
    return vals[k-1]; 
  };


  var kthSmallest = function (root, k) {
    const stack = [];
    let count = 1;
    let node = root;
  
    while (node || stack.length) {
      while (node) {
        stack.push(node);
        node = node.left;
      }
      node = stack.pop();
      if (count === k) return node.val;
      else count++;
      node = node.right;
    }
  };

  const kthSmallest = (root, k) => {
    let n = 0;
    let res;
    const inorder = (root) => {
      if (!root) return;
      inorder(root.left);
      if (n++ < k) res = root.val;
      inorder(root.right);
    };
    inorder(root);
    return res;
  };

  var isValidBST = function(root) {     
    function helper(node, lower, upper){
        if (!node){
            return true}
        if (lower < node.val && node.val < upper){
            return helper(node.left, lower, node.val) && helper(node.right, node.val, upper)}
        else{
            return false}}       
    return helper( node=root, lower=-Infinity, upper=Infinity )

};

var isValidBST = function(root) {     
    function* inorder(node) {
        if (node) {
            yield* inorder(node.left);
            yield node.val;
            yield* inorder(node.right);
        }
    }
    let generator = inorder(root)
    let output = generator.next()
    while (!output.done){
        let check = output.value
        output = generator.next()
        if (!output.done && check >= output.value){
            return false}}
    return true}

var isValidBST = function(root, min=null, max=null) {
    if (!root) return true;
    if (min && root.val <= min.val) return false;
    if (max && root.val >= max.val) return false;
    return isValidBST(root.left, min, root) && isValidBST(root.right, root, max);
};

var numIslands = function(grid) {
    const directions = [[1, 0], [-1, 0], [0, -1], [0, 1]];
    let count = 0
    for(let i = 0; i < grid.length; i++){
        for(let j = 0; j < grid[0].length; j++){
            if (grid[i][j] == '1'){
                dfs(grid,i,j)
                count  += 1}}}
    return count

function dfs(grid,i,j){
    
    grid[i][j] = 0
    for (let [dr, dc] of directions) {
        let r = i + dr
        let c = j + dc
        if (0 <= r && r < grid.length && 0 <= c && c < grid[0].length && grid[r][c]=='1'){
            dfs(grid,r,c)}}}

};

const numIslands =  (grid) => {
	let count = 0 
	
	for(let row = 0; row < grid.length; row++){
	for(let col = 0; col < grid[row].length; col ++){
	if(grid[row][col] == '1'){
		count ++
		explore(row,col, grid)
            }
        }
    }
    return count
}




function explore(row, col, grid){
    
    
     if (row < 0 || col < 0 || row >= grid.length  
         || col >= grid[row].length || grid[row][col] === '0')  {
        return
    }
    
    
    
    grid[row][col]='0'
    
	
	
	
	explore(row, col+1, grid)   
    
	explore(row, col-1, grid)  
    
	explore(row+1, col, grid) 
    
	explore(row-1, col, grid)   

}

var numIslands = function(grid) {
    let count = 0
    for(let i = 0; i < grid.length; i++){
        for(let j = 0; j < grid[0].length; j++){
            if (grid[i][j] == '1'){
                grid[i][j] = '0'
                helper(grid,i,j)
                count  += 1}}}
    return count

    function helper(grid,i,j){
        let queue = [[i,j]]
        while (queue.length > 0){
            [x,y] = queue.shift()
            let directions = [[x+1,y],[x,y+1],[x-1,y],[x,y-1]]
            for (let[i,j] of directions ){
                if (0 <= i  && i< grid.length && 0 <= j && j < grid[0].length && grid[i][j] == '1'){

                    grid[i][j] = '0'
                    queue.push([i,j])}}}}
};

var numIslands = function(grid) {
    let count = 0
    for(let i = 0; i < grid.length; i++){
        for(let j = 0; j < grid[0].length; j++){
            if (grid[i][j] == '1'){
                grid[i][j] = '0'
                helper(grid,i,j)
                count  += 1}}}
    return count

    function helper(grid, i, j) {
        let queue = [[i, j]];
        let start = 0;
        let end = 1;  

        while (start < end) {
            let [x, y] = queue[start];
            start++;

            let directions = [[x+1, y], [x, y+1], [x-1, y], [x, y-1]];
            for (let [i, j] of directions) {
                if (0 <= i && i < grid.length && 0 <= j && j < grid[0].length && grid[i][j] == '1') {
                    grid[i][j] = '0';
                    queue.push([i, j]);
                    end++;
                }
            }
        }
    }

};

var numIslands = function(grid) {
    let count = 0
    for(let i = 0; i < grid.length; i++){
        for(let j = 0; j < grid[0].length; j++){
            if (grid[i][j] == '1'){
                grid[i][j] = '0'
                helper(grid,i,j)
                count  += 1}}}
    return count

    function helper(grid,i,j){
        let queue = [[i,j]]
        while (queue.length > 0){
            const next = []
            for (let [x,y] of queue){
                let directions = [[x+1,y],[x,y+1],[x-1,y],[x,y-1]]
                for (let[i,j] of directions ){
                    if (0 <= i  && i< grid.length && 0 <= j && j < grid[0].length && grid[i][j] == '1'){

                        grid[i][j] = '0'
                        next.push([i,j])}
                        }
                
            }
            queue = next
        }
    }
};

var numIslands = function(grid) {
    let count = 0;
    
    function depthSearch(x, y) {
        if (grid[x][y] === '1') {
            grid[x][y] = '0';
        } else {
            return;
        }

        if (x < grid.length - 1) {
            depthSearch(x+1, y);
        }
        
        if (y < grid[x].length - 1) {
            depthSearch(x, y+1);
        }
        
        if (x > 0) {
            depthSearch(x-1, y);
        }
        
        if (y > 0) {
            depthSearch(x, y-1);
        }
    }
    
    for (let i = 0; i < grid.length; i++) {
        for (let j = 0; j < grid[i].length; j++) {
            if (grid[i][j] === '1') {
                count++;
                depthSearch(i, j);
            }
        }
    }
    
    return count;
};

var solve = function(board) {

    function depthSearch(x, y) {
                
        if (board[x][y] === 'O') {
            board[x][y] = 'S';
        } else {
            return;
        }

        if (x < board.length - 1) {
            depthSearch(x+1, y);
        }
        
        if (y < board[x].length - 1) {
            depthSearch(x, y+1);
        }
        
        if (x > 0) {
            depthSearch(x-1, y);
        }
        
        if (y > 0) {
            depthSearch(x, y-1);
        }
    }

    for (let i = 0; i < board.length; i++) {
        if (board[i][0] === 'O') {
            depthSearch(i, 0);
        }
        if (board[i][board[0].length-1] === 'O') {
            depthSearch(i, board[0].length-1);
        }
    }
    for (let i = 0; i < board[0].length; i++) {
        if (board[0][i] === 'O') {
            depthSearch(0, i);
        }
        if (board[board.length-1][i] === 'O') {
            depthSearch(board.length -1, i);
        }
    }
      for (let i = 0; i < board.length; i++) {
        for (let j = 0; j < board[i].length; j++) {
            if (board[i][j] === 'O') {
                board[i][j] = 'X'
            }
            else if (board[i][j] === 'S'){
                board[i][j] = 'O'
            }
        }
    } 
};

var solve = function(board) {
    if(board.length ==0) return null 
    
    for(var i=0;i<board.length;i++){
        for(var j=0;j<board[0].length;j++){
            if(board[i][j] == 'O' && (i==0 || i==board.length-1 || j==0 || j==board[0].length-1)){
                  dfs(board,i,j)
               }
        }
    }
    
    for(var i=0;i<board.length;i++){
        for(var j=0;j<board[0].length;j++){
            if(board[i][j]=='W'){
                  board[i][j]='O'
               }
            else {
                    board[i][j]='X'
                    }
        }
    }
    
    return board
};

  function dfs(board,i,j){
      if(i<0 || j<0 || i>=board.length || j >=board[0].length || board[i][j]=='X' || board[i][j]=='W'){
            return 
         }
      board[i][j]='W';
      dfs(board,i+1,j)
      dfs(board,i-1,j)
      dfs(board,i,j+1)
      dfs(board,i,j-1)
      return 
  }

  const dirs = [[0, 1], [1, 0], [-1, 0], [0, -1]];
var solve = function(board) {
    const rows = board.length, cols = board[0].length;
    let visited = Array.from({ length: rows }, () => new Array(cols).fill(0));
    let queue = [];
    
    for(let i = 0; i < rows; i++){
        
        if(board[i][0] === 'O'){
            queue.push([i, 0]);
        }
        if(board[i][cols-1] === 'O'){
            queue.push([i, cols-1]);
        }
    }
    for(let i = 0; i < cols; i++){
        
        if(board[0][i] === 'O'){
            queue.push([0, i]);
        }
        if(board[rows-1][i] === 'O'){
            queue.push([rows-1, i]);
        }
    }
    
    bfs(queue, visited, board);
    
    for(let i = 0; i < rows; i++){
        for(let j = 0; j < cols; j++){
            if(!visited[i][j]){
                board[i][j] = 'X';
            }
        }
    }
};

function bfs(queue, visited, board){
    
    while(queue.length){
        
        let cell = queue.pop();
        let x = cell[0], y = cell[1];
        
        visited[x][y] = true;
        
        for(let d of dirs){
            let nx = x+d[0], ny = y+d[1];
            if (nx < 0 || nx >= board.length || ny < 0 || ny >= board[0].length || visited[nx][ny] || board[nx][ny] != 'O') {
                continue;
            }
            queue.unshift([nx, ny]);
        }
    }
}
var cloneGraph = function(node) {
    return helper(node, {})
    

    function helper(node, visited){
        if (!node){
            return null}
        
        let newNode = new Node(node.val)
        visited[node.val] = newNode
        
        for(let adjNode of node.neighbors){
            if (! (adjNode.val in visited)){
                newNode.neighbors.push(helper(adjNode, visited))}
            else{
                newNode.neighbors.push(visited[adjNode.val])}}
        
        return newNode}
    
};

function cloneGraph(graph) {
    var map = {};
    return traverse(graph);
  
    function traverse(node) {
      if (!node) return node;
      if (!map[node.val]) {
        map[node.val] = new Node(node.val);
        map[node.val].neighbors = node.neighbors.map(traverse);
      }
      return map[node.val];
    }
  }

  var cloneGraph = function(node) {
    
    let start = node; 
    if (start === null) return null;
    
    const vertexMap = new Map(); 
    
    
    
    const queue = [start]
    vertexMap.set(start, new Node(start.val)); 
    
        
    while (queue.length > 0) {
        
        const currentVertex = queue.shift(); 
        
        for (const neighbor of currentVertex.neighbors) {
          
            if (!vertexMap.has(neighbor)) {
                                vertexMap.set(neighbor, new Node(neighbor.val))
                queue.push(neighbor); 
            }
            
                        vertexMap.get(currentVertex).neighbors.push(vertexMap.get(neighbor)); 
        }
    }
   return vertexMap.get(start); 
    
};

var calcEquation = function(equations, values, queries) {
    let output = []
    let equations_graph = {}
    for(let i = 0; i < equations.length; i++){
        let [a, b] = equations[i]
        if (!(a in equations_graph)){
            equations_graph[a] = {}}
        if (!(b in equations_graph)){
            equations_graph[b] = {}}

        equations_graph[a][b] = values[i]
        equations_graph[b][a] = 1.0 / values[i]}

    for (let query of queries){
        let[a, b] = query
        if (!(a in equations_graph) || (!( b in equations_graph))){
            output.push(-1.0)
            continue}
        output.push(dfs(a,b,equations_graph,1.0, new Set()))}
    return output

    function dfs(a,b,equations_graph,weight,visit){
        if (a === b){
            return weight}
        visit.add(a)
        for (let next_node in equations_graph[a]) {
            let ratio = equations_graph[a][next_node];
            if (!(visit.has(next_node))){
                let result = dfs(next_node,b,equations_graph,weight*ratio,visit)
                if (result !== -1.0){
                    return result}}}
        return -1.0}
    
};


var calcEquation = function(equations, values, queries) {
    const adjList = new Map();
    
    
    for (let i = 0; i < equations.length; i++) {
        adjList.set(equations[i][0], []);
        adjList.set(equations[i][1], []);
    }

    
    for (let i = 0; i < equations.length; i++) {
        const u = equations[i][0];
        const v = equations[i][1];
        const weight = values[i];
        
        
        adjList.get(u).push([v,weight]);
        
        
        adjList.get(v).push([u, 1/weight]);
    }
    
    
    const res = [];
    
    for (let i = 0; i < queries.length; i++) {
        
        const src = queries[i][0];
        
        
        const dest = queries[i][1];
        const seen = new Set();
        const val = dfs(adjList, src, src, dest, 1, seen);
        
        
        if (val === false) {
            res.push(-1);
        } else {
            res.push(val);
        }
    }
    
    return res;
}


var dfs = function(adjList, src, curr, dest, quotient, seen) {
    
    if (!adjList.has(dest) || !adjList.has(src)) {
        return -1;
    }
    
    
    if (src === dest) {
        return 1;
    }
    
    
    if (curr === dest) {
        return quotient;
    }
    
    seen.add(curr);
    
    const neighbors = adjList.get(curr);
    
    for (let i = 0; i < neighbors.length; i++) {
        if (seen.has(neighbors[i][0])) {
            continue;
        }
        const val = dfs(adjList, src, neighbors[i][0], dest, quotient * neighbors[i][1], seen);
        
        
        if (val !== false) return val;
    }
    
    
    
    
    return false;
}

var calcEquation = function(equations, values, queries) {
    let output = [];
    let equations_graph = {};
    
    for(let i = 0; i < equations.length; i++){
        let [a, b] = equations[i];
        if (!(a in equations_graph)){
            equations_graph[a] = {};}
        if (!(b in equations_graph)){
            equations_graph[b] = {};}

        equations_graph[a][b] = values[i];
        equations_graph[b][a] = 1 / values[i];
    }

    for (let query of queries){
        let [a, b] = query;
        if (!(a in equations_graph) || !(b in equations_graph)){
            output.push(-1.0);
            continue;
        }
        output.push(bfs(a, b, equations_graph));
    }
    return output;

    function bfs(start, end, graph) {
        let queue = [{node: start, weight: 1.0}];
        let visited = new Set();

        while (queue.length > 0) {
            let {node, weight} = queue.shift();

            if (node === end) {
                return weight;
            }

            visited.add(node);

            for (let next_node in graph[node]) {
                if (!visited.has(next_node)) {
                    queue.push({
                        node: next_node,
                        weight: weight * graph[node][next_node]
                    });
                }
            }
        }

        return -1.0;
    }
};


function buildAdjacencyList(n, edgesList){
    let adjList = Array.from({ length: n }, () => []);
    for( let [c1, c2] of edgesList){
        adjList[c2].push(c1)}
    return adjList}

function topoBFS(numNodes, edgesList){
            let adjList = buildAdjacencyList(numNodes, edgesList)

            let inDegrees = new Array(numNodes).fill(0);
            for (let [v1, v2] of edgesList){
                inDegrees[v1] += 1}
    
            let queue = []
            for (let v = 0; v < numNodes; v++){
                if (inDegrees[v] == 0){
                    queue.push(v)}}
    
            let count = 0

            let topoOrder = []
    
            while (queue.length > 0){

                let v = queue.shift()

                topoOrder.push(v)

                count += 1

                for (let des of adjList[v]){
                    inDegrees[des] -= 1
                    if (inDegrees[des] == 0){
                        queue.push(des)}}}
    
            if (count != numNodes){
                return null}
            else{
                return topoOrder}}

var canFinish = function(numCourses, prerequisites) {
    if (topoBFS(numCourses, prerequisites)){
        return true
    }
    return false
    
};

var canFinish = function(numCourses, prerequisites) {
    const order = [];
    const queue = [];
    const graph = new Map();
    const indegree = Array(numCourses).fill(0);
  
    for (const [e, v] of prerequisites) {
      
      if (graph.has(v)) {
        graph.get(v).push(e);
      } else {
        graph.set(v, [e]);
      }
      
      indegree[e]++;
    }
  
    for (let i = 0; i < indegree.length; i++) {
      if (indegree[i] === 0) queue.push(i);
    }
  
    while (queue.length) {
      const v = queue.shift();
      if (graph.has(v)) {
        for (const e of graph.get(v)) {
          indegree[e]--;
          if (indegree[e] === 0) queue.push(e);
        }
      }
      order.push(v);
    }
  
    return numCourses === order.length;
  };


  
let visiting;   
let visited;  
let graph;

var canFinish = function(numCourses, prerequisites) {
    graph = new Map();
    visiting = new Set();
    visited = new Set();
    
    for(let [v, e] of prerequisites){
        if(graph.has(v)){
            let edges = graph.get(v);
            edges.push(e);
            graph.set(v,edges);
        }else{
            graph.set(v,[e]);
        }
    }
    
    for(const [v,e] of graph){
        if(DFS(v)){
            return false; 
        }
    }
    
    return true;
}

var DFS = function(v){
    visiting.add(v);
    let edges = graph.get(v);   
    
    if(edges){
        
       for(let e of edges){
            if(visited.has(e)){ 
                continue;
            }

            if(visiting.has(e)){ 
                return true;
            }

            if(DFS(e)){ 
                return true;
            }
        } 
    }   
    
    visiting.delete(v); 
    visited.add(v);
    return false;
}

var findOrder = function(numCourses, prerequisites) {
    const order = [];
    const queue = [];
    const graph = Array.from({ length: numCourses }, () => []);
    const indegree = Array(numCourses).fill(0);
  
    for (const [e, v] of prerequisites) {
      graph[v].push(e)
      
      indegree[e]++;
    }
  
    for (let i = 0; i < indegree.length; i++) {
      if (indegree[i] === 0) queue.push(i);
    }
  
    while (queue.length) {
      const v = queue.shift();
      for (const e of graph[v]) {
        indegree[e]--;
        if (indegree[e] === 0) queue.push(e);
        }
      order.push(v);
    }
  
    if (numCourses === order.length){
        return order
    }
    else{
        return []
    }
  };

const findOrder = (numCourses, prerequisites) => {
const inDegrees = Array(numCourses).fill(0);
for (const [v] of prerequisites) {
    inDegrees[v]++;
}

const q = [];
for (let i = 0; i < inDegrees.length; i++) {
    const degree = inDegrees[i];
    if (degree === 0) q.push(i);
}

const res = [];
while (q.length) {
    const u0 = q.shift();
    numCourses--;
    res.push(u0);
    for (const [v, u] of prerequisites) {
    if (u === u0) {
        inDegrees[v]--;
        if (inDegrees[v] === 0) q.push(v);
    }
    }
}
return numCourses === 0 ? res : [];
};



var findOrder = function(numCourses, prerequisites) {
    
    graph = new Map();
    visited = new Array(numCourses).fill(0);
    stack = new Array();
    
    for(let [v, e] of prerequisites){
        if(graph.has(v)){
            let values = graph.get(v);
            values.push(e);
            graph.set(v, values)
        } else {
            graph.set(v, [e])
        }
    }
    
    for(let i = 0; i < numCourses; i++){
        if(visited[i] == 0 && DFS(i)) return [];
    }
    
    return stack;
}


function DFS(index){
    
    visited[index] = 1;
    let edges = graph.get(index);
    
    if(edges){
        for(let e of edges){
            if(visited[e] == 1) return true;
            if(visited[e] == 0 && DFS(e)) return true
        }  
    }

    visited[index] = 2;
    stack.push(index)
    return false
}


var snakesAndLadders = function(board) {
    
    board.reverse();
    for (let i = 1; i < board.length; i += 2) {
      board[i].reverse();
    }
  
    const arr = [null].concat(...board);  
    
    const n = arr.length - 1;
    const queue = [1];
    const seen = new Set([1]);
    let ct = 0;
    
    while (queue.length > 0) {
      const lenQ = queue.length;
      
      
      for (let i = 0; i < lenQ; i++) {
        const cur = queue.shift();
        
        if (cur === n) {
          return ct;
        }
        
        
        for (let j = cur + 1; j <= Math.min(cur + 6, n); j++) {
          const nxt = arr[j]+1 !== 0 ? arr[j] : j;
          
          if (seen.has(nxt)) {
            continue;
          }
          
          seen.add(nxt);
          queue.push(nxt);
        }
      }
      
      ct += 1;
    }
    
    return -1;
  }
  

var snakesAndLadders = function(board) {
    let n = board.length
    function label_to_position(label){
        let r = Math.floor((label-1) / n)
        let c = (label - 1) %n
        if (r % 2 === 0){
            return [n-1-r, c]}
        else{
            return [n-1-r, n-1-c]}}
        
    let seen = new Set()
    let queue = []
    queue.push([1, 0])
    while (queue.length != 0){
        let [label, step] = queue.shift()
        let [r, c] = label_to_position(label)
        if (board[r][c] != -1){
            label = board[r][c]}
        if (label == n*n){
            return step}
        for (let x = 1; x <= 6; x++ ){
            let new_label = label + x
            if (new_label <= n*n && !seen.has(new_label)){
                seen.add(new_label)
                queue.push([new_label, step+1])}}}
    return -1}

var snakesAndLadders = function(board) {
    const N = board.length;
    const getLoc = (pos) => {
        let row = Math.floor((pos - 1) / N);
        let col = (pos - 1) % N;
        col = (row % 2) === 1 ? N - col - 1 : col;
        row = N - row - 1;
        return [row,col];
    }
    const q = [1];
    const v = {'1': 0};
    while(q.length) {
        const n = q.shift();
        if(n === N*N) return v[n];
        for(let i = n+1; i <= Math.min(n+6, N*N); i++) {
        const [r, c] = getLoc(i);
        const next = board[r][c] === -1 ? i : board[r][c];
        if(v[next] === undefined) {
            q.push(next);
            v[next] = v[n] + 1;
        }
        }
    }
    
    return -1;
    };

var minMutation = function(startGene, endGene, bank) {
    let bank_set = new Set(bank) 
    if (!bank_set.has(endGene)){
        return -1}
    
    let queue = [[startGene, 0]]
    let visited = new Set([startGene]) 

    while (queue.length){ 
        [current_gene, mutations] = queue.shift()
        if (current_gene === endGene){
            return mutations}

        for (let i = 0; i <8; i++){
            for (c of ['A', 'C', 'G', 'T']){
                let next_gene = current_gene.substring(0, i) + c + current_gene.substring(i + 1)
                if (bank_set.has(next_gene) && !visited.has(next_gene)){
                    visited.add(next_gene)
                    queue.push([next_gene, mutations + 1])}}}}

    return -1
    
};

var minMutation = function(start, end, bank) {
    if (!bank.includes(end)) {
        return -1;
    }
    
    if (start === end) {
        return 0;
    }
    
    let queue = [[start, [start]]];
    let steps = 0;
    
    const getOneStepMutations = (current, visited) => {
        return bank.filter((mutation) => {
            if (visited.includes(mutation)) {
                return false;
            }
            
            let difference = 0;
            
            for (let i = 0; i < 8; i++) {
                if (current[i] !== mutation[i]) {
                    if (++difference > 1) {
                        return false;
                    }
                }
            }
            
            return difference === 1;
        });
    };
    
    while (queue.length > 0) {
        let next = [];
        
        for (const [current, visited] of queue) {
            const mutations = getOneStepMutations(current, visited);
            
            for (const mutation of mutations) {
                if (mutation === end) {
                    return steps + 1;
                }
                
                visited.push(mutation);
                next.push([mutation, visited]);
                visited.pop();
            }
        }
        
        queue = next;
        steps++;
    }
    
    return -1;
};

const getDifferent = (string1, string2) => {
    let output = 0;
    for (let i = 0 ; i < 8; i++) {
        if (string1[i] !== string2[i]) output++;
    }    
    return output;
}

var minMutation = function(start, end, bank) {
    if (bank.filter(item => item === end)?.length < 1) return -1;
        
    const loop = (current, currentBank) => {
        let output = Infinity;
        if (getDifferent(end, current) === 1) return 1;

        for (let i = 0; i < currentBank.length; i++) {
            if (getDifferent(current, currentBank[i]) === 1) {
                const newBank = currentBank.filter(item => item !== currentBank[i]);
                const newCount = loop(currentBank[i], newBank);
                if (newCount !== - 1) output = Math.min(output, newCount);
            }
        }

        return output !== Infinity ? output + 1 : -1;
    }
    return loop(start, bank);
};

var ladderLength = function(beginWord, endWord, wordList) {
    let L = beginWord.length
    let all_combo_dict = {}
    for (let word of wordList){
        for (let i = 0; i <L;i++){
            let new_word = word.substring(0,i) + "*" + word.substring(i+1)
            if (new_word in all_combo_dict){
                all_combo_dict[new_word].push(word)}
            else{
                all_combo_dict[new_word] = [word]
            }}}
    let queue = [[beginWord, 1]]
    let visited = new Set()
    visited.add(beginWord)
    while (queue.length){
        let [current_word, level] = queue.shift()
        for (let i = 0; i <L; i++){
            let intermediate_word = current_word.substring(0,i) + "*" + current_word.substring(i+1)
            if (all_combo_dict[intermediate_word]) {
                for (let word of all_combo_dict[intermediate_word]){
                    if (word === endWord){
                        return level + 1}
                    if (!visited.has(word)){
                        visited.add(word)
                        queue.push([word, level + 1])}}}}}
    return 0
    
};

var ladderLength = function(beginWord, endWord, wordList) {
    function difference(word1, word2) {
           let oneDiff = false;
           for (let i = 0; i < word1.length; i++) {
               if (word1[i] !== word2[i]) {
                   if (oneDiff) {
                       return false;
                   }
                   oneDiff = true;
               }
           }
           return true;
       }
     
       const seen = new Set([beginWord]);
       const queue = [beginWord];
       let sequence = 1;
     
       while (queue.length > 0) {
           sequence++;
           let queueSize = queue.length;
           
           for (let i = 0; i < queueSize; i++) {
               const word = queue.shift();
               
               for (const change of wordList) {
                   if (!seen.has(change) && difference(word, change)) {
                       if (change === endWord) {
                           return sequence;
                       }
                       queue.push(change);
                       seen.add(change);
                   }
               }
           }
       }
       return 0;
   }


   var ladderLength = function(beginWord, endWord, wordList) {
    const wordSet = new Set(wordList)
    let queue = [beginWord];
    let steps = 1;
    
    while(queue.length) {
        const next = [];
        
        
        for(let word of queue) {
            if(word === endWord) return steps;
            
            
            for(let i = 0; i < word.length; i++) {
                
                
                for(let j = 0; j < 26; j++) {
                    const newWord = word.slice(0, i) + String.fromCharCode(j + 97) + word.slice(i+1);
                    
                    
                    if(wordSet.has(newWord)) {
                        next.push(newWord);
                        wordSet.delete(newWord);
                    }
                }
            }
        }
        queue = next
        steps++;
    }
    return 0;    
};

class TrieNode {
    constructor() {
        this.children = new Map();
        this.end = false;
    }}

var Trie = function() {
    this.root = new TrieNode()
    
};

Trie.prototype.insert = function(word) {
    node = this.root
    for (let char of word){
        if (!node.children.has(char)){
            node.children.set(char, new TrieNode())
        }
        node = node.children.get(char)}
    node.end = true
    
    
};

Trie.prototype.search = function(word) {
    node = this.root
    for (let char of word){
        if (!node.children.has(char)){
            return false
        }
        node = node.children.get(char)}
    return node.end
    
    
};

Trie.prototype.startsWith = function(prefix) {
    node = this.root
    for (let char of prefix){
        if (!node.children.has(char)){
            return false
        }
        node = node.children.get(char)}
    return true
    
    
};

class Trie {
    constructor() {
      this.root = {};
    }
  
    insert(word) {
      let node = this.root;
      for (let c of word) {
        if (node[c] == null) node[c] = {};
        node = node[c];
      }
      node.isWord = true;
    }
  
    traverse(word) {
      let node = this.root;
      for (let c of word) {
        node = node[c];
        if (node == null) return null;
      }
      return node;
    }
  
    search(word) {
      const node = this.traverse(word);
      return node != null && node.isWord === true;
    }
  
    startsWith(prefix) {
      return this.traverse(prefix) != null;
    }
  }

  class TrieNode {
    constructor() {
      this.children = {};
      this.isEnd = false;
    }
  }
  
  class WordDictionary {
    constructor() {
      this.root = new TrieNode();
    }
  
    addWord(word) {
      let cur = this.root;
      for (const c of word) {
        if (!cur.children[c]) {
          cur.children[c] = new TrieNode();
        }
        cur = cur.children[c];
      }
      cur.isEnd = true;
    }
  
    search(word) {
      let nodes = [this.root];
      for (const c of word) {
        if (nodes.length === 0) {
          return false;
        }
        const all_nodes = [];
        for (const node of nodes) {
          if (node.children[c]) {
            all_nodes.push(node.children[c]);
          } else if (c === ".") {
            all_nodes.push(...Object.values(node.children));
          }
        }
        nodes = all_nodes;
      }
      for (const node of nodes) {
        if (node.isEnd) {
          return true;
        }
      }
      return false;
    }
  }
  

  class TrieNode {
    constructor() {
      this.children = {};
      this.isEnd = false;
    }
  }
  
  class WordDictionary {
    constructor() {
      this.root = new TrieNode();
    }
  
    addWord(word) {
      let cur = this.root;
      for (const c of word) {
        if (!cur.children[c]) {
          cur.children[c] = new TrieNode();
        }
        cur = cur.children[c];
      }
      cur.isEnd = true;
    }
  
    search(word) {
      function dfs(node, index){
          if (index === word.length){
              return node.isEnd}
              
          if( word[index] === "."){
              for (let child of Object.values(node.children)){
                  if (dfs(child, index+1)){
                      return true}}}
                  
          if (node.children[word[index]]){
              return dfs(node.children[word[index]], index+1)}
          
          return false}
  
      return dfs(this.root, 0)
    }
  }
  

  var WordDictionary = function() {
    this.trie = {};
};

WordDictionary.prototype.addWord = function(word) {
    let root = this.trie;
    for (let i=0;i<word.length;i++) {
        if (root[word[i]]==null) root[word[i]] = {};
        root = root[word[i]];
    }
    root.isEnd = true;  
};

WordDictionary.prototype.search = function(word) {
    return this.dfs(word, 0, this.trie);
};

WordDictionary.prototype.dfs = function(word, index, node) {
    if (index == word.length) return node.isEnd == true;
    
    if (word[index] == '.') {
        for (let key in node) {
            if (this.dfs(word, index + 1, node[key])) return true;
        }
        
    } else {
        if (node[word[index]]!=null) {
            return this.dfs(word, index + 1, node[word[index]]);
        }
    }
    return false;
}




var findWords = function(board, word_list) {
    let word_dict = new WordDictionary()
    for (let word of word_list){
        word_dict.addWord(word)}

    let words = []
    
    function dfs(row,col, node){
        
        let letter = board[row][col]
        let cur = node[letter]
        board[row][col] = ""
        
        if ('isEnd' in cur){
            words.push(cur['isEnd'])
            delete cur["isEnd"]}

        if (row > 0 &&  board[row-1][col] in cur){
            dfs(row-1,col,cur)}
        if (col > 0 && board[row][col-1] in cur){
            dfs(row,col-1,cur)}
        if (row < board.length-1 && board[row+1][col] in cur){
            dfs(row+1,col,cur)}
        if (col < board[0].length-1 && board[row][col+1] in cur){
            dfs(row,col+1,cur)}
        
        board[row][col] = letter
        if (!Object.keys(cur).length){
            delete node[letter]}}

    for(let i =0 ; i < board.length; i++){
        for (let j =0 ; j < board[0].length; j++){
            if(board[i][j] in word_dict.trie){
                dfs(i,j,word_dict.trie)}}}
    return words

};

var WordDictionary = function() {
this.trie = {};
};

WordDictionary.prototype.addWord = function(word) {
let root = this.trie;
for (let i=0;i<word.length;i++) {
    if (root[word[i]]==null) root[word[i]] = {};
    root = root[word[i]];
}
root.isEnd = word;  
};



const findWords = (board, words) => {
    const dirs = [[-1, 0], [0, 1], [1, 0], [0, -1]];
    let res = [];
  
    const buildTrie = () => {
      const root = {};
      for (const w of words) {
        let node = root;
        for (const c of w) {
          if (node[c] == null) node[c] = {};
          node = node[c];
        }
        node.word = w;
      }
      return root;
    };
  
    const search = (node, x, y) => {
      if (node.word != null) {
        res.push(node.word);
        node.word = null; 
      }
  
      if (x < 0 || x >= board.length || y < 0 || y >= board[0].length) return;
      if (node[board[x][y]] == null) return;
  
      const c = board[x][y];
      board[x][y] = '#'; 
      for (const [dx, dy] of dirs) {
        const i = x + dx;
        const j = y + dy;
        search(node[c], i, j);
      }
      board[x][y] = c; 
    };
  
    const root = buildTrie();
    for (let i = 0; i < board.length; i++) {
      for (let j = 0; j < board[0].length; j++) {
        search(root, i, j);
      }
    }
    return res;
  };

  var letterCombinations = function(digits) {
    let letters = ["abc","def","ghi","jkl","mno","pqrs","tuv","wxyz"]
    let output = []
    for (let digit of digits){
        let temp = output
        output = []
        for (let letter of letters[digit-2]){
            if (!temp.length){
                output.push(letter)}
            else{
                for (let combination of temp){
                    output.push(combination + letter)}}}}
    return output

    
};

var letterCombinations = function(digits) {
    const letters = ["abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"];
    
    if (digits.length === 0) return [];
    
    return digits.split('').map(digit => letters[digit - 2].split('')).reduce((acc, curr) => {
      return acc.map(a => curr.map(c => a + c)).flat();
    });
  };

  const L = {'2':"abc",'3':"def",'4':"ghi",'5':"jkl",
     '6':"mno",'7':"pqrs",'8':"tuv",'9':"wxyz"}

var letterCombinations = function(D) {
    let len = D.length, ans = []
    if (!len) return []
    const dfs = (pos, str) => {
        if (pos === len) ans.push(str)
        else {
            let letters = L[D[pos]]
            for (let i = 0; i < letters.length; i++)
                dfs(pos+1,str+letters[i])
        }
    }
    dfs(0,"")
    return ans
};

var combine = function(n, k) {
    function backtrack(first = 1, curr = []){
        if (curr.length == k){
            output.push(curr.slice())
            return}
        for(let i = first;i < n+1;i++){
            curr.push(i)
            backtrack(i + 1, curr)
            curr.pop()}}
    output = []
    backtrack()
    return output
    
};



function* generateCombinations(elems, num) {
    const total = elems.length;
    if (num > total) {
      return;
    }
    let currIndices = Array.from({ length: num }, (_, i) => i);
    
    while (true) {
      yield currIndices.map(i => elems[i]);
      
      let idx;
      for (idx = num - 1; idx >= 0; idx--) {
        if (currIndices[idx] !== idx + total - num) {
          break;
        }
      }
      
      if (idx < 0) {
        return;
      }
      
      currIndices[idx]++;
      for (let j = idx + 1; j < num; j++) {
        currIndices[j] = currIndices[j - 1] + 1;
      }
    }
  }
  
  function combine(n, k) {
    const elems = Array.from({ length: n }, (_, i) => i + 1);
    const result = [];
    
    for (const combination of generateCombinations(elems, k)) {
      result.push(combination);
    }
    
    return result;
  }

var permute = function(nums) {
    let full = nums.length
    let visited = new Set()
    function backtrack(curr = []){
        if (curr.length == full){
            output.push(curr.slice())
            return}
        for(let num of nums){
            if(!visited.has(num)){
                visited.add(num)
                curr.push(num)
                backtrack(curr)
                curr.pop()
                visited.delete(num)}}}
    output = []
    backtrack()
    return output
    
};

var permute = function(letters) {
    let res = [];
    dfs(letters, [], Array(letters.length).fill(false), res);
    return res;
}

function dfs(letters, path, used, res) {
    if (path.length == letters.length) {
        
        res.push(Array.from(path));
        return;
    }
    for (let i = 0; i < letters.length; i++) {
        
        if (used[i]) continue;
        
        path.push(letters[i]);
        used[i] = true;
        dfs(letters, path, used, res);
        
        path.pop();
        used[i] = false;
    }
}

var combinationSum = function(candidates, target) {
    let res = []
    candidates.sort((a, b) => a - b)
    dfs(target, 0, [])
    return res
        
    function dfs(target, index, path){
        if (target < 0){
            return}
        if (target == 0){
            res.push(path)
            return }
        for (let i = index ; i <candidates.length;i++){
            dfs(target-candidates[i], i, [...path, candidates[i]])}}

    
};

var combinationSum = function(candidates, target) {
    
    const dp = Array.from({ length: target + 1 }, () => []);
    
    
    for (let c of candidates) {
      
      for (let i = c; i <= target; i++) {
        
        if (i === c) {
          dp[i].push([c]);
        }
        
        
        for (let comb of dp[i - c]) {
          dp[i].push([...comb, c]);
        }
      }
    }
    
    
    return dp[target];
  }

function combinationSum(candidates, target) {
    var buffer = [];
    var result = [];
    search(0, target);
    return result;
  
    function search(startIdx, target) {
      if (target === 0) return result.push(buffer.slice());
      if (target < 0) return;
      if (startIdx === candidates.length) return;
      buffer.push(candidates[startIdx]);
      search(startIdx, target - candidates[startIdx]);
      buffer.pop();
      search(startIdx + 1, target);
    }
  }

  var combinationSum = function(candidates, target) {
    let res = [];
    candidates.sort((a, b) => a - b);
    dfs(target, 0, []);
    return res;
        
    function dfs(target, index, path) {
        if (target < 0) {
            return true;  
        }
        if (target === 0) {
            res.push(path);
            return;
        }
        for (let i = index; i < candidates.length; i++) {
            if (dfs(target - candidates[i], i, [...path, candidates[i]])) {
                break;  
            }
        }
    }
};

var totalNQueens = function(n) {
    let locations = new Set()
    let used = new Set()
    let count = 0

    function dfs(col){
        if (used.size === n){
            count += 1}
        for (let row = 0; row < n; row++){
            if (!used.has(row) && safe(row,col)){
                locations.add(`${row},${col}`)
                used.add(row)
                dfs(col+1)
                used.delete(row)
                locations.delete(`${row},${col}`)}}}
        
    function safe(row, col){
        for( coords of locations){
            let [r, c] = coords.split(',').map(Number)
            if( Math.abs(row - r) === Math.abs(col - c)){
                return false}}
        return true}
    dfs(0)
    return count
    
};


var totalNQueens = function(n) {

    let visited_cols= new Set()

    visited_diagonals= new Set()

    visited_antidiagonals= new Set()
    
    let count = 0

    function backtrack(r){
        if (r===n){
            count += 1
            return }     
        for (let c = 0;  c < n; c++){
            if (!(visited_cols.has(c) || visited_diagonals.has(r-c) || visited_antidiagonals.has(r+c))){
                visited_cols.add(c)
                visited_diagonals.add(r-c)
                visited_antidiagonals.add(r+c)
                backtrack(r+1)
                visited_cols.delete(c)
                visited_diagonals.delete(r-c)
                visited_antidiagonals.delete(r+c)}}}
                    
    backtrack(0)
    return count
};

var totalNQueens = function(N) {
    let ans = 0
    
    const place = (i, vert, ldiag, rdiag) => {
        if (i === N) ans++
        else for (let j = 0; j < N; j++) {
            let vmask = 1 << j, lmask = 1 << (i+j), rmask = 1 << (N-1-i+j)
            if (vert & vmask || ldiag & lmask || rdiag & rmask) continue
            place(i+1, vert | vmask, ldiag | lmask, rdiag | rmask)
        }
    }

    place(0,0,0,0)
    return ans
};

var totalNQueens = function(n) {
    const cols = new Set(),
          hills = new Set(),
          dales = new Set();
        const isSafe = (row, col) => !(cols.has(col) || hills.has(row - col) || dales.has(row + col));
    
        const placeQueen = (row, col) => {
        cols.add(col), hills.add(row - col), dales.add(row + col);
    }
    
        const removeQueen = (row, col) => {
        cols.delete(col), hills.delete(row - col), dales.delete(row + col);
    }
    
        const backtrackQueen = (row, count) => {
        if (row === n) {
            return ++count;
        }
        for (let col = 0; col < n; col++) {
            if (isSafe(row, col)) {
                placeQueen(row, col);
                count = backtrackQueen(row + 1, count);
                removeQueen(row, col);
            }
        }
        return count;
    }
    return backtrackQueen(0, 0);
};

var generateParenthesis = function(n) {
    function dfs(left, right, s){
        if( s.length=== n * 2){
            res.push(s)
            return }

        if (left < n){
            dfs(left + 1, right, s + '(')}

        if (right < left){
            dfs(left, right + 1, s + ')')}}

    let res = []
    dfs(0, 0, '')
    return res
            
    
};

function generateParenthesis(n) {
    const m = 2 * n;
    const output = [];
    let val = 0;
    let length = 0;
    const curr = [];

    function dfs() {
        if (length === m) {
            output.push(curr.join(''));
            return;
        }
        if (val > 0) {
            curr.push(')');
            val -= 1;
            length += 1;
            dfs();
            curr.pop();
            val += 1;
            length -= 1;
        }
        if (val < m - length) {
            curr.push('(');
            val += 1;
            length += 1;
            dfs();
            curr.pop();
            val -= 1;
            length -= 1;
        }
    }

    dfs();
    return output;
}

const generateParenthesis = (n) => {
    const res = [];
  
    const go = (l, r, s) => { 
      if (l > r) return; 
  
      if (l === 0 && r === 0) {
        res.push(s);
        return;
      }
  
      if (l > 0) go(l - 1, r, s + '(');
      if (r > 0) go(l, r - 1, s + ')');
    };
  
    go(n, n, '');
    return res;
  };


  var exist = function(board, word) {
    let row_length = board.length-1
    let col_length = board[0].length-1
    let word_length = word.length - 1
    let word_found = false

    if (word.length > (row_length+1)*(col_length+1)){
        return false}
    
    let boardDic = {}
	for (let i = 0; i < row_length+1; i++){
		for (let j = 0; j < col_length+1; j++){
			if(boardDic[board[i][j]]){
                boardDic[board[i][j]] += 1
            }
            else{
                boardDic[board[i][j]] = 1
            }}}

    for (let i = 0; i < word_length+1; i++){
			if(boardDic[word[i]]){
                boardDic[word[i]] -= 1
                if (boardDic[word[i]] == -1){
                    return false
                }
            }
            else{
                return false
            }}
        

    function dfs(row,col,index){
        if (board[row][col] !== word[index]){
            return}
        if (index === word_length){
            word_found = true
            return}
        let temp = board[row][col]
        board[row][col] = ""
        if (row > 0 ){
            dfs(row-1,col,index+1)}
        if (col > 0 && !word_found){
            dfs(row,col-1,index+1)}
        if (row < row_length && !word_found){
            dfs(row+1,col,index+1)}
        if (col < col_length && !word_found ){
            dfs(row,col+1,index+1)}
        board[row][col] = temp}
	for (let i = 0; i < row_length+1; i++){
		for (let j = 0; j < col_length+1; j++){
            if (!word_found){
                dfs(i,j,0)}}}
    return word_found
        
    
};

const exist = (board, word) => {
    if (board.length === 0) return false;
  
    const h = board.length;
    const w = board[0].length;
    const dirs = [[-1, 0], [0, 1], [1, 0], [0, -1]];
  
    const go = (x, y, k) => {
      if (board[x][y] !== word[k]) return false;
      if (k === word.length - 1) return true;
  
      board[x][y] = '*'; 
      for (const [dx, dy] of dirs) {
        const i = x + dx;
        const j = y + dy;
        if (i >= 0 && i < h && j >= 0 && j < w) {
          if (go(i, j, k + 1)) return true;
        }
      }
      board[x][y] = word[k]; 
      return false;
    };
  
    for (let i = 0; i < h; i++) {
      for (let j = 0; j < w; j++) {
        if (go(i, j, 0)) return true;
      }
    }
  
    return false;
  };


var exist = function(board, word) {
    return isValidByAvailableSymbols(board, word) && isValidForSegmentsOfWord(board, word)
        ? checkPhrase(board, word)
        : false;
};

const isValidByAvailableSymbols = (board, word) => {
    let wordCounter = {};

    for (let i = 0; i < board.length; i ++) {
        for (let j = 0; j < board[i].length; j ++) {
            wordCounter[board[i][j]] = (wordCounter[board[i][j]] || 0) + 1;
        }
    }

    for (let i = 0; i < word.length; i ++) {
        if (wordCounter[word[i]] !== undefined && wordCounter[word[i]] >= 0) {
            wordCounter[word[i]] --;
        } else {
            return false;
        }
    }

    return true;
};

const isValidForSegmentsOfWord = (board, word) => {
    const thirdLength = Math.floor(word.length / 3);
    const twoThirdLength = Math.floor(word.length * 2 / 3);

    if (word.length > 5) {
        const firstSegment = word.substring(0, thirdLength);
        const secondSegment = word.substring(thirdLength, twoThirdLength);
        const thirdSegment = word.substring(twoThirdLength);

        if (!checkPhrase(clone(board), firstSegment) || 
            !checkPhrase(clone(board), secondSegment) || 
            !checkPhrase(clone(board), thirdSegment)) {
            return false;
        }
    }

    return true;
}

const checkPhrase = (board, word) => {
    for (let i = 0; i < board.length; i ++) {
        for (let j = 0; j < board[i].length; j ++) {
            if (dfs(board, word, i, j)) {
                return true;
            }
        }
    }

    return false;
};

const clone = (a) => JSON.parse(JSON.stringify(a));

const directionMap = [0, 1, 0, -1, 0];
const dfs = (board, word, i, j, cursor = 0) => {
    
    
    if (cursor === word.length) {
        return true;
    }

    
    if (board[i]?.[j] !== word[cursor] || board[i][j] === -1) {
        return false;
    }

    
    board[i][j] = -1;

    
    for (let k = 0; k < 4; k ++) {
        if (dfs(board, word, i + directionMap[k], j + directionMap[k + 1], cursor + 1)) {
            return true;
        }
    }

    board[i][j] = word[cursor];

    return false;
};


var sortedArrayToBST = function(nums) {
    if (!nums.length){
        return null}
    let mid = Math.floor(nums.length / 2)
    let parent = new TreeNode(nums[mid])
    parent.left = sortedArrayToBST(nums.slice(0,mid))
    parent.right = sortedArrayToBST(nums.slice(mid+1))
    return parent
    
};



var sortList = function(head) {
    if(!head){
        return null}
    let length = 0
    let dummy = new ListNode()
    dummy.next = head
    while (head){
        length +=1 
        head = head.next}
    
    function divide(start,length){
        if (length === 1){
            start.next = null
            return start}
        let mid = Math.floor(length/2)
        let prev  = null
        let curr = start
        for (let i = 0 ; i < mid; i++){
            prev = curr
            curr = curr.next}
        prev.next = null
        
        return merge(divide(start,mid), divide(curr, length - mid))}

    
    function merge(list1,list2){
        let new_dummy = new ListNode()
        let current = new_dummy
        while (list1 && list2){
            if (list1.val <= list2.val){
                current.next = list1
                list1 = list1.next}
            else{
                current.next = list2
                list2 = list2.next}
            current = current.next}
        current.next = list1 || list2
        return new_dummy.next}
    


    return divide(dummy.next,length)
    
    
};


var sortList = function(head) {
    if(head==null){
        return null;
    }
    let ptr=head;
    let arr=[];
    while(ptr){
        arr.push(ptr.val);
        ptr=ptr.next;
    }
    arr.sort((a,b)=>a-b);
    let n = new ListNode(arr[0]);
    head=n;
    let temp=head;
    for(let i=1;i<arr.length;i++){
        let n1 = new ListNode(arr[i]);
        temp.next=n1;
        temp=temp.next;       
    }
    return head;
};


var merge = function (l1, l2) {
    let dummy = new ListNode(0);
    let tail = dummy;

    while (l1 && l2) {
        if (l1.val <= l2.val) {
            tail.next = l1;
            l1 = l1.next;
        } else {
            tail.next = l2;
            l2 = l2.next;
        }
        tail = tail.next;
    }

    if (l1)
        tail.next = l1;
    else if (l2)
        tail.next = l2;

    return dummy.next;
};

var sortList = function (head) {
    if (!head || !head.next)
        return head;

    let slow = head;
    let fast = head.next;
    while (fast && fast.next) {
        slow = slow.next;
        fast = fast.next.next;
    }

    let mid = slow.next;
    slow.next = null;

    let l1 = sortList(head);
    let l2 = sortList(mid);

    return merge(l1, l2);
};


var sortList = function(head) {
    if(head==null){
        return null;
    }
    let ptr=head;
    let arr=[];
    while(ptr){
        arr.push(ptr.val);
        ptr=ptr.next;
    }
    arr.sort((a,b)=>a-b);
    let n = new ListNode(arr[0]);
    head=n;
    let temp=head;
    for(let i=1;i<arr.length;i++){
        let n1 = new ListNode(arr[i]);
        temp.next=n1;
        temp=temp.next;       
    }
    return head;
};



var sortList = function(head){
    if (head === null || head.next === null)
        return head;

    const getLength = function (head) {
        let length = 0;
        let curr = head;
        while (curr) {
            length++;
            curr = curr.next;
        }
        return length;
    };

    const split = function (head, step) {
        if (head === null)
            return null;

        for (let i = 1; i < step && head.next; i++) {
            head = head.next;
        }

        const right = head.next;
        head.next = null;
        return right;
    };

    const merge = function (left, right, tail) {
        let curr = tail;
        while (left && right) {
            if (left.val < right.val) {
                curr.next = left;
                left = left.next;
            } else {
                curr.next = right;
                right = right.next;
            }
            curr = curr.next;
        }

        curr.next = left ? left : right;
        while (curr.next)
            curr = curr.next;

        return curr;
    };

    const length = getLength(head);
    const dummy = new ListNode(0);
    dummy.next = head;

    let step = 1;
    while (step < length) {
        let curr = dummy.next;
        let tail = dummy;

        while (curr) {
            const left = curr;
            const right = split(left, step);
            curr = split(right, step);

            tail = merge(left, right, tail);
        }

        step *= 2;
    }

    return dummy.next;
}



function sortList(head) {
    const buffer_size = 8;
    const dummy = new ListNode(0);
    dummy.next = head;
  
    let steps = 1;
    while (true) {
      let prev = dummy;
      let remaining = prev.next;
  
      let num_loops = 0;
      while (remaining) {
        num_loops++;
  
        const sublists = Array(buffer_size).fill(null);
        const sublists_tail = Array(buffer_size).fill(null);
        for (let i = 0; i < buffer_size; i++) {
          sublists[i] = remaining;
          let substeps = steps;
          while (substeps && remaining) {
            substeps--;
            sublists_tail[i] = remaining;
            remaining = remaining.next;
          }
          if (sublists_tail[i]) {
            sublists_tail[i].next = null;
          }
        }
  
        let num_sublists = buffer_size;
        while (num_sublists > 1) {
          const subdummy = new ListNode();
          for (let i = 0; i < num_sublists; i += 2) {
            let subprev = subdummy;
            subprev.next = null;
            while (sublists[i] && sublists[i + 1]) {
              if (sublists[i].val <= sublists[i + 1].val) {
                subprev.next = sublists[i];
                sublists[i] = sublists[i].next;
              } else {
                subprev.next = sublists[i + 1];
                sublists[i + 1] = sublists[i + 1].next;
              }
              subprev = subprev.next;
            }
  
            if (sublists[i]) {
              subprev.next = sublists[i];
              sublists_tail[Math.floor(i / 2)] = sublists_tail[i];
            } else {
              subprev.next = sublists[i + 1];
              sublists_tail[Math.floor(i / 2)] = sublists_tail[i + 1];
            }
  
            sublists[Math.floor(i / 2)] = subdummy.next;
          }
          num_sublists = Math.floor(num_sublists / 2);
        }
  
        prev.next = sublists[0];
        prev = sublists_tail[0];
      }
  
      steps *= buffer_size;
  
      if (num_loops <= 1) {
        return dummy.next;
      }
    }
  }
  


  var construct = function(grid) {
            
    function allSame(i, j, w){
        for (let x = i; x < i + w; x++){
            for (let y = j; y < j + w; y++){
                if (grid[x][y] !== grid[i][j]){
                    return false}}}
        return true}

    function helper(i, j, w){
        if (allSame(i, j, w)){
            return new Node(grid[i][j], true)}
        let new_length = Math.floor(w/2)
        let node = new Node(true, false)
        node.topLeft = helper(i, j, new_length)
        node.topRight = helper( i, j + new_length, new_length)
        node.bottomLeft = helper( i + new_length, j, new_length)
        node.bottomRight = helper( i + new_length, j + new_length, new_length)
        return node}

    return helper(0, 0, grid.length)

};

var construct = function(grid) {
    const helper = (row, col, length) => {
      if (length === 1) {
        return new Node(grid[row][col], true);
      }
      const newLength = Math.floor(length / 2);
      const TL = helper(row, col, newLength);
      const TR = helper(row, col + newLength, newLength);
      const BL = helper(row + newLength, col, newLength);
      const BR = helper(row + newLength, col + newLength, newLength);

      if (TL.isLeaf && TL.val === TR.val && TL.val === BL.val && TL.val === BR.val) {
        return TL;
      }
      return new Node(-1, false, TL, TR, BL, BR);
    };

    return helper(0, 0, grid.length);
  }


  var construct = function(grid) {
    const len = grid.length;
    
    function split(matrix) {
        const [[rowStart, colStart], [rowEnd, colEnd]] = matrix;
        const halfWidth = (rowEnd - rowStart) / 2 
        const midRow = rowStart + halfWidth;
        const midCol = colStart + halfWidth;
        
        const topLeft = [[rowStart, colStart], [midRow, midCol]];
        const topRight = [[rowStart, midCol], [midRow, colEnd]];
        const bottomLeft = [[midRow, colStart], [rowEnd, midCol]];
        const bottomRight = [[midRow, midCol], [rowEnd, colEnd]]
        
        return {topLeft, topRight, bottomLeft, bottomRight}
    }
    
    function recurse(matrix) {
        const [[rowStart, colStart], [rowEnd, colEnd]] = matrix;
        
        if(rowEnd - rowStart === 1) return new Node(grid[rowStart][colStart], true)
        
        const {topLeft, topRight, bottomLeft, bottomRight} = split(matrix);
        
        const nodeTL = recurse(topLeft);
        const nodeTR = recurse(topRight);
        const nodeBL = recurse(bottomLeft);
        const nodeBR = recurse(bottomRight);
        
        
        
        if(nodeTL.isLeaf && nodeTR.isLeaf && nodeBL.isLeaf && nodeBR.isLeaf && 
           (nodeTL.val === nodeTR.val && nodeTR.val === nodeBL.val && nodeBL.val === nodeBR.val))  {
            return new Node(nodeTL.val, true);
        }
        return new Node(false, false, nodeTL, nodeTR, nodeBL, nodeBR);
    }
    return recurse([[0, 0], [len, len]])
};

var mergeKLists = function(lists) {
    function merge2lists(head1,head2){
        let dummy = new ListNode()
        let output = dummy
        while (head1 && head2){
            if (head1.val <= head2.val){
                output.next = head1
                head1 = head1.next}
            else if (head2.val < head1.val){
                output.next = head2
                head2 = head2.next}
            output = output.next}
        output.next = head1 || head2
        return dummy.next}

    let length = lists.length
    while (length >1){
        for(let  i = 0; i < length; i +=2 ){
            if (i+1 === length){
                lists[Math.floor(i/2)] = merge2lists(lists[i],null)
                continue}
            lists[Math.floor(i/2)] = merge2lists(lists[i],lists[i+1])}
        length = Math.ceil(length/2)}
    return length !== 0 ? lists[0] : null;

};



function merge(left, right) {
    const dummy = new ListNode(-1);
    let temp = dummy;
  
    while (left && right) {
      if (left.val < right.val) {
        temp.next = left;
        temp = temp.next;
        left = left.next;
      } else {
        temp.next = right;
        temp = temp.next;
        right = right.next;
      }
    }
  
    while (left) {
      temp.next = left;
      temp = temp.next;
      left = left.next;
    }
  
    while (right) {
      temp.next = right;
      temp = temp.next;
      right = right.next;
    }
  
    return dummy.next;
  }
  
  function mergeSort(lists, start, end) {
    if (start === end) {
      return lists[start];
    }
  
    const mid = start + Math.floor((end - start) / 2);
    const left = mergeSort(lists, start, mid);
    const right = mergeSort(lists, mid + 1, end);
  
    return merge(left, right);
  }
  
  function mergeKLists(lists) {
    if (!lists || lists.length === 0) {
      return null;
    }
    return mergeSort(lists, 0, lists.length - 1);
  }


  var maxSubArray = function(nums) {
    maxSum = -Infinity 
    currentSum = 0
    
    for (let num of nums){
        currentSum += num
        
        if (currentSum > maxSum){
            maxSum = currentSum}
        
        if (currentSum < 0){
            currentSum = 0}}
    
    return maxSum
    
};


var maxSubArray = function(nums) {
    
    let maxSum = nums[0];
    
    for (let i = 1; i < nums.length; i++) {
        
        
        
        
        nums[i] = Math.max(0, nums[i - 1]) + nums[i];
        
        if (nums[i] > maxSum)
            maxSum = nums[i];
    }
    return maxSum;      
};


var maxSubArray = function(nums) {
    let Output = nums[0]
    let current = nums[0]
    for (i =1 ; i < nums.length; i++){
        current = Math.max(current + nums[i], nums[i])
        if (current > Output){
            Output = current}}
    return Output}


function maxSubArray(A) {
    var prev = 0;
    var max = -Number.MAX_VALUE;
    
    for (var i = 0; i < A.length; i++) {
        prev = Math.max(prev + A[i], A[i]);
        max = Math.max(max, prev);
    }
    return max;
    }


var maxSubarraySumCircular = function(nums) {
    let total_sum = 0
    let curr = 0
    let max_sum = -Infinity
    let flag = 1
    let ans = -Infinity

    for (let i of nums){
        if (i >= 0){
            flag = 0
            break}
        ans = Math.max(ans, i)}

    if (flag){
        return ans}

    for (let i of nums){
        total_sum += i
        curr += i
        max_sum = Math.max(max_sum, curr)
        if (curr < 0){
            curr = 0}}
    let min_sum = Infinity
    curr = 0

    for (let i of nums){
        curr += i
        min_sum = Math.min(min_sum, curr)
        if (curr > 0){
            curr = 0}}

    let ans2 = total_sum - min_sum
    return Math.max(max_sum, ans2)

};

var maxSubarraySumCircular = function(A) {
    let maxSum, max, minSum, min, total
    maxSum = max = minSum = min= total = A[0]
    
    for(let i=1;i<A.length;i++){
      const n = A[i]
      max = Math.max(n, n+max)
      maxSum = Math.max(max, maxSum)
      min = Math.min(n, n+min)
      minSum = Math.min(min, minSum)
      total += n
    }
    return maxSum > 0 ? Math.max(maxSum, total - minSum) : maxSum
  };

  var searchInsert = function(nums, target) {
    let low = 0, high = nums.length
    while( low < high){
        let mid = Math.floor((low + high)/2)
        if (target > nums[mid]){
            low = mid + 1}
        else{
            high = mid}}
    return low
    
};

var searchInsert = function(nums, target) {
    let low = 0, high = nums.length - 1
    while( low <=  high){
        let mid = Math.floor((low + high)/2)
        if (target == nums[mid]){
            return mid
        }
        else if (target > nums[mid]){
            low = mid + 1}
        else{
            high = mid-1}}
    return low
    
};

function searchInsert(nums, target) {
    return binarySearch(nums, target, 0, nums.length - 1);
};


function binarySearch(array, target, start, end) {
	 
	 
	 
	 
    if (start > end) return start;
    
    const midPoint = Math.floor((start + end)/2);
    
	
    if (array[midPoint] === target) return midPoint;
    
	
    if (array[midPoint] > target) return binarySearch(array, target, start, midPoint - 1);
    
    if (array[midPoint] < target) return binarySearch(array, target, midPoint + 1, end);
}

var searchInsert = function(nums, target) {
    for(let i =0;i<nums.length;i++){
        if(nums[i] >= target)   return i;
    }
    return nums.length;
};

var searchMatrix = function(matrix, target) {
    if (!matrix.length){
        return false}
    let m =matrix.length, n = matrix[0].length
    let left= 0,right = m * n - 1

    while (left <= right){
        let mid = Math.floor((left + right) / 2)
        let mid_row = Math.floor(mid/n), mid_col = mid%n

        if (matrix[mid_row][mid_col] == target){
            return true}
        else if (matrix[mid_row][mid_col] < target){
            left = mid + 1}
        else{
            right = mid - 1}}

    return false
    
};

var searchMatrix = function(matrix, target) {
    let m = matrix.length;
    let n = matrix[0].length;
    let left = 0, right = m * n - 1;

    while (left <= right) {
        let mid = Math.floor((left + right) / 2);
        let mid_val = matrix[Math.floor(mid / n)][mid % n];

        if (mid_val === target)
            return true;
        else if (mid_val < target)
            left = mid + 1;
        else
            right = mid - 1;
    }
    return false;
};

function searchMatrix(matrix, target) {
    let low = 0, high = matrix.length - 1;
    
    while (low < high) {
        let mid = Math.floor((low + high) / 2);
        if (target === matrix[mid][0]) {
            return true;
        } else if (target > matrix[mid][0]) {
            if (target < matrix[mid + 1][0]) {
                low = mid;
                high = mid;
            } else {
                low = mid + 1;
            }
        } else {
            high = mid - 1;
        }
    }
    
    if (matrix[low][0] === target) {
        return true;
    }
    
    let row = low;
    low = 0, high = matrix[0].length - 1;
    while (low <= high) {
        let mid = Math.floor((low + high) / 2);
        if (target === matrix[row][mid]) {
            return true;
        } else if (target > matrix[row][mid]) {
            low = mid + 1;
        } else {
            high = mid - 1;
        }
    }
    
    return false;
}

function searchMatrix(matrix, target) {
    if (!matrix.length || !matrix[0].length) return false;
  
    let row = 0;
    let col = matrix[0].length - 1;
  
    while (col >= 0 && row <= matrix.length - 1) {
      if (matrix[row][col] === target) return true;
      else if (matrix[row][col] > target) col--;
      else if (matrix[row][col] < target) row++;
    }
  
    return false;
  }

  var findPeakElement = function(nums) {

    let left = 0, right =  nums.length-1
    
    while (left < right){
        let mid = Math.floor((left + right) / 2)
        if (nums[mid] > nums[mid+1]){
            right = mid}
        else{
            left = mid + 1}}
            
    return left
};

var findPeakElement = function(nums) {

    let low = 0, high =  nums.length-1

    while (low <= high){
        let mid = Math.floor((low + high) / 2)
        let middle = nums[mid]
        let compare_r = (mid + 1 <= nums.length - 1) ? nums[mid + 1] : -Infinity;
        let compare_l = (mid - 1 >= 0) ? nums[mid - 1] : -Infinity;

        if (middle < compare_r){
            low = mid + 1}
        else if (middle < compare_l){
            high = mid - 1}
        else{
            return mid}}}

var findPeakElement = function(nums) {
    for(let i = 0; i < nums.length; i++) {
        if(nums[i] > nums[i+1]) return i;
    }
    return nums.length-1;
};

function search(nums, target) {
    let left = 0;
    let right = nums.length - 1;

    while (left <= right) {
        const mid = Math.floor((left + right) / 2);

        if (nums[mid] === target) {
            return mid;
        }

        
        if (nums[left] <= nums[mid]) {
            if (nums[left] <= target && target < nums[mid]) {
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        
        else {
            if (nums[mid] < target && target <= nums[right]) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
    }

    return -1;
}

var search = function(nums, target) {
    let left = 0 ,right = nums.length-1 
    let end= nums[right]
    
    while (left < right){
        let mid = Math.floor((left + right) / 2)
        if (nums[mid] < end){
            right = mid  }
        else{
            left = mid + 1}}
            
    let pivot = left

    left = 0, right = nums.length-1
    let length =  nums.length
    
    while (left <= right){
        mid = Math.floor((left + right) / 2)
        let real_mid = (mid + pivot)%length
        if (target === nums[real_mid]){
            return real_mid}
        else if ( target > nums[real_mid]){
            left = mid + 1}
        else{
            right = mid - 1}}
    return -1

};


var search = function (nums, target) {
    let lo = 0;
    let hi = nums.length - 1;
    
    while (lo < hi) {
      let mid = Math.floor((lo + hi) / 2);
      if (nums[mid] > nums[hi]) {
        lo = mid + 1;
      } else {
        hi = mid;
      }
    }
    
    if (target === nums[lo]) {
      return lo;
    }
    if (target < nums[lo]) {
      return -1;
    }
    if (lo > 0 && target > nums[lo - 1]) {
      return -1;
    }
    if (target <= nums[nums.length - 1]) {
      hi = nums.length - 1;
    } else {
      hi = lo - 1; 
      lo = 0;
    }
    
    while (lo <= hi) {
      let mid = Math.floor((lo + hi) / 2);
      if (nums[mid] === target) {
        return mid;
      }
      if (target > nums[mid]) {
        lo = mid + 1;
      } else {
        hi = mid - 1;
      }
    }
    return -1;
  };

  function searchRange(nums, target) {
    let low = 0, high = nums.length - 1;
    let left = -1, right = -1;

    while (low <= high) {
        let mid = Math.floor((low + high) / 2);
        if (target === nums[mid]) {
            let compare_r = (mid + 1 <= nums.length - 1) ? nums[mid + 1] : Number.NEGATIVE_INFINITY;
            if (nums[mid] === compare_r) {
                low = mid + 1;
            } else {
                right = mid;
                break;
            }
        } else if (target > nums[mid]) {
            low = mid + 1;
        } else {
            high = mid - 1;
        }
    }

    low = 0, high = nums.length - 1;
    while (low <= high) {
        let mid = Math.floor((low + high) / 2);
        if (target === nums[mid]) {
            let compare_l = (mid - 1 >= 0) ? nums[mid - 1] : Number.NEGATIVE_INFINITY;
            if (nums[mid] === compare_l) {
                high = mid - 1;
            } else {
                left = mid;
                break;
            }
        } else if (target > nums[mid]) {
            low = mid + 1;
        } else {
            high = mid - 1;
        }
    }

    return [left, right];
}

var searchRange = function(nums, target) {
            
    function search(x){
        let lo = 0, hi = nums.length         
        while (lo < hi){
            let mid = Math.floor((lo + hi) / 2)
            if (nums[mid] < x){
                lo = mid+1}
            else{
                hi = mid     }}               
        return lo}
    
    let low = search(target)
    let high = search(target+1)-1
    
    if (low <= high){
        return [low, high]}
            
    return [-1, -1]

};

var searchRange = function(N, T) {
    const find = (target, arr, left=0, right=arr.length) => {
        while (left <= right) {
            let mid = left + right >> 1
            if (arr[mid] < target) left = mid + 1
            else right = mid - 1
        }
        return left
    } 
    let Tleft = find(T, N)
    if (N[Tleft] !== T) return [-1,-1]
    return [Tleft, find(T+1, N, Tleft) - 1]
};

var findMin = function(nums) {
    let lo = 0;
    let hi = nums.length - 1;

    while (lo < hi) {
      let mid = Math.floor((lo + hi) / 2);
      if (nums[mid] > nums[hi]) {
        lo = mid + 1;
      } else {
        hi = mid;
      }
    }
    return nums[lo]
};

var findMedianSortedArrays = function(nums1, nums2) {
    if (nums1.length > nums2.length){
        [nums1, nums2] = [nums2, nums1]}
    let low = 0 ,high= nums1.length
    let len1 =nums1.length, len2 = nums2.length
    let Total = (len1 + len2)
    while (low <= high){
        let split1 = Math.floor((low + high) / 2)
        let split2 =  Math.floor((Total+1)/2) - split1

        let maxX = split1 == 0 ? -Infinity: nums1[split1 - 1]
        let maxY = split2 == 0 ? -Infinity: nums2[split2 - 1]

        let minX = split1 == len1 ? Infinity: nums1[split1]
        let minY = split2 == len2 ? Infinity: nums2[split2]

        if (maxX <= minY && maxY <= minX){
            return ( (len1 + len2) % 2 == 0 ? (Math.max(maxX, maxY) + Math.min(minX, minY) )/ 2 :  Math.max(maxX, maxY) ) }
        else if (minX < maxY){
            low = split1 + 1}
        else{
            high = split1 - 1}}
    
};

function findMedianSortedArrays(nums1, nums2) {
    
    const merged = nums1.concat(nums2);
    
    
    merged.sort((a, b) => a - b);
    
    
    const total = merged.length;
    
    if (total % 2 === 1) {
        
        return parseFloat(merged[Math.floor(total / 2)]);
    } else {
        
        const middle1 = merged[Math.floor(total / 2) - 1];
        const middle2 = merged[Math.floor(total / 2)];
        return (parseFloat(middle1) + parseFloat(middle2)) / 2.0;
    }
}

function findMedianSortedArrays(nums1, nums2) {
    const n = nums1.length;
    const m = nums2.length;
    let i = 0;
    let j = 0;
    let m1 = 0;
    let m2 = 0;
    
    
    for (let count = 0; count < Math.floor((n + m) / 2) + 1; count++) {
        m2 = m1;
        if (i < n && j < m) {
            if (nums1[i] > nums2[j]) {
                m1 = nums2[j];
                j++;
            } else {
                m1 = nums1[i];
                i++;
            }
        } else if (i < n) {
            m1 = nums1[i];
            i++;
        } else {
            m1 = nums2[j];
            j++;
        }
    }
    
    
    if ((n + m) % 2 === 1) {
        return parseFloat(m1);
    } else {
        const ans = parseFloat(m1) + parseFloat(m2);
        return ans / 2.0;
    }
}


class MinHeap {
    constructor() {
        this.heap = [];
    }
    push(val) {
        this.heap.push(val);
        this.bubbleUp();
    }
    pop() {
        const max = this.heap[0];
        const end = this.heap.pop();
        if (this.heap.length > 0) {
            this.heap[0] = end;
            this.bubbleDown();
        }
        return max;
    }
    peek() {
        return this.heap[0];
    }
    bubbleUp() {
        let idx = this.heap.length - 1;
        const element = this.heap[idx];
        while (idx > 0) {
            let parentIdx = Math.floor((idx - 1) / 2);
            let parent = this.heap[parentIdx];
            if (element >= parent) break;
            this.heap[parentIdx] = element;
            this.heap[idx] = parent;
            idx = parentIdx;
        }
    }
    bubbleDown() {
        let idx = 0;
        const length = this.heap.length;
        const element = this.heap[0];
        while (true) {
            let leftChildIdx = 2 * idx + 1;
            let rightChildIdx = 2 * idx + 2;
            let leftChild, rightChild;
            let swap = null;
            if (leftChildIdx < length) {
                leftChild = this.heap[leftChildIdx];
                if (leftChild < element) {
                    swap = leftChildIdx;
                }
            }
            if (rightChildIdx < length) {
                rightChild = this.heap[rightChildIdx];
                if (
                    (swap === null && rightChild < element) || 
                    (swap !== null && rightChild < leftChild)
                ) {
                    swap = rightChildIdx;
                }
            }
            if (swap === null) break;
            this.heap[idx] = this.heap[swap];
            this.heap[swap] = element;
            idx = swap;
        }
    }
}

var findKthLargest = function(nums, k) {
    let heap = new MinHeap()
    for (let i = 0; i < k; i++){
        heap.push(nums[i])
    }
    for (let i = k; i < nums.length; i++){
        if (nums[i] > heap.peek()){
            heap.pop()
            heap.push(nums[i])}
    }
    return heap.peek()
};


var findKthLargest = function(nums, k) {
    let heap = new MinHeap();
    for (let i = 0; i < k; i++) {
        heap.push(nums[i]);
    }
    for (let i = k; i < nums.length; i++) {
        heap.push(nums[i]);
        heap.pop();  
    }
    return heap.peek();
};


var findKthLargest = function(nums, k) {
    nums.sort((a, b) => b - a);
    return nums[k-1];
};


var findKthLargest = function(nums, k) {
    const partition = (left, right, pivotIndex) => {
        const pivot = nums[pivotIndex];
        [nums[pivotIndex], nums[right]] = [nums[right], nums[pivotIndex]];
        let storedIndex = left;
        for (let i = left; i < right; i++) {
            if (nums[i] < pivot) {
                [nums[storedIndex], nums[i]] = [nums[i], nums[storedIndex]];
                storedIndex++;
            }
        }
        [nums[right], nums[storedIndex]] = [nums[storedIndex], nums[right]];
        return storedIndex;
    };
    
    let left = 0, right = nums.length - 1;
    while (true) {
        const pivotIndex = left + Math.floor(Math.random() * (right - left + 1));
        const newPivotIndex = partition(left, right, pivotIndex);
        if (newPivotIndex === nums.length - k) {
            return nums[newPivotIndex];
        } else if (newPivotIndex > nums.length - k) {
            right = newPivotIndex - 1;
        } else {
            left = newPivotIndex + 1;
        }
    }
};


class MinHeap {
    constructor() {
        this.heap = [];
    }
    length() {
        return this.heap.length
    }
    push(val) {
        this.heap.push(val);
        this.bubbleUp();
    }
    pop() {
        const max = this.heap[0];
        const end = this.heap.pop();
        if (this.heap.length > 0) {
            this.heap[0] = end;
            this.bubbleDown();
        }
        return max;
    }
    peek() {
        return this.heap[0];
    }
    bubbleUp() {
        let idx = this.heap.length - 1;
        const element = this.heap[idx];
        while (idx > 0) {
            let parentIdx = Math.floor((idx - 1) / 2);
            let parent = this.heap[parentIdx];
            if (element >= parent) break;
            this.heap[parentIdx] = element;
            this.heap[idx] = parent;
            idx = parentIdx;
        }
    }
    bubbleDown() {
        let idx = 0;
        const length = this.heap.length;
        const element = this.heap[0];
        while (true) {
            let leftChildIdx = 2 * idx + 1;
            let rightChildIdx = 2 * idx + 2;
            let leftChild, rightChild;
            let swap = null;
            if (leftChildIdx < length) {
                leftChild = this.heap[leftChildIdx];
                if (leftChild < element) {
                    swap = leftChildIdx;
                }
            }
            if (rightChildIdx < length) {
                rightChild = this.heap[rightChildIdx];
                if (
                    (swap === null && rightChild < element) || 
                    (swap !== null && rightChild < leftChild)
                ) {
                    swap = rightChildIdx;
                }
            }
            if (swap === null) break;
            this.heap[idx] = this.heap[swap];
            this.heap[swap] = element;
            idx = swap;
        }
    }
}

var findMaximizedCapital = function(k, w, profits, capital) {
    let n = profits.length
    let projects = capital.map((cap, i) => [cap, profits[i]]);
    projects.sort((a,b) => a[0]-b[0])
    let i = 0
    let maximizeCapital = new MinHeap()
    while (k){
        while (i < n && projects[i][0] <= w){
            maximizeCapital.push(-projects[i][1])
            i += 1}
        if (!maximizeCapital.length()){
            break}
        w -= maximizeCapital.pop()
        k -= 1}
    return w
    
};

var findMaximizedCapital = function(k, w, profits, capital) {
    let projects = [];
    let heap = new MaxHeap();

    for(let i = 0; i < profits.length; i++){
        projects.push([profits[i], capital[i]]);
    }

    projects.sort((a, b) => a[1] - b[1]);
    let x = 0;

    while(projects[x] && projects[x][1] <= w){
        heap.add(projects[x][0]);
        x++;
    }

    while(heap.values.length > 0 && k > 0){
        w += heap.extractMax();
        k--;

        while(projects[x] && projects[x][1] <= w){
            heap.add(projects[x][0]);
            x++;
        }
    }

    return w;
};



class MaxHeap {
    constructor() {
        this.values = [];
    }

    
    parent(index) {
        return Math.floor((index - 1) / 2);
    }

    
    leftChild(index) {
        return (index * 2) + 1;
    }

    
    rightChild(index) {
        return (index * 2) + 2;
    }

    
    isLeaf(index) {
        return (
            index >= Math.floor(this.values.length / 2) && index <= this.values.length - 1
        )
    }

    
    swap(index1, index2) {
        [this.values[index1], this.values[index2]] = [this.values[index2], this.values[index1]];
    }

    heapifyDown(index) {
        
        if (!this.isLeaf(index)) {

            
            let leftChildIndex = this.leftChild(index),
                rightChildIndex = this.rightChild(index),

                
                largestIndex = index;

            
            if (this.values[leftChildIndex] > this.values[largestIndex]) {
                
                largestIndex = leftChildIndex;
            }

            
            if (this.values[rightChildIndex] >= this.values[largestIndex]) {
                
                largestIndex = rightChildIndex;
            }

            
            if (largestIndex !== index) {
                
                this.swap(index, largestIndex);
                
                this.heapifyDown(largestIndex);
            }
        }
    }

    heapifyUp(index) {
        let currentIndex = index,
            parentIndex = this.parent(currentIndex);

        
        while (currentIndex > 0 && this.values[currentIndex] > this.values[parentIndex]) {
            
            this.swap(currentIndex, parentIndex);
            
            currentIndex = parentIndex;
            parentIndex = this.parent(parentIndex);
        }
    }

    add(element) {
        
        this.values.push(element);
        
        this.heapifyUp(this.values.length - 1);
    }

    
    peek() {
        return this.values[0];
    }

    
    extractMax() {
        if(this.values.length === 1) return this.values.pop();

        if (this.values.length < 1) return 'heap is empty';

        
        const max = this.values[0];
        const end = this.values.pop();
        
        this.values[0] = end;
        
        this.heapifyDown(0);

        
        return max;
    }

    buildHeap(array) {
        this.values = array;
        
        for(let i = Math.floor(this.values.length / 2); i >= 0; i--){
            this.heapifyDown(i);
        }
    }

    print() {
        let i = 0;
        while (!this.isLeaf(i)) {
            console.log("PARENT:", this.values[i]);
            console.log("LEFT CHILD:", this.values[this.leftChild(i)]);
            console.log("RIGHT CHILD:", this.values[this.rightChild(i)]);
            i++;
        }      
    }
}


var findMaximizedCapital = function(k, res, profits, capital) {     
    let maxheap = new MaxPriorityQueue({priority: v => v[0]});
    let minheap = new MinPriorityQueue({priority: v => v[1]});
    for(let i = 0; i < profits.length; i++){                  
        maxheap.enqueue([profits[i],capital[i]]);
    }
    while(k&&maxheap.size()){
        let [value,limit] = maxheap.dequeue().element;        
        if(limit<=res) k--, res+=value;    
        else minheap.enqueue([value,limit]);
        while(minheap.size()&&minheap.front().priority<=res){ 
        let [value,limit] = minheap.dequeue().element;    
        maxheap.enqueue([value,limit]);    
    }}
    return res;
};

var kSmallestPairs = function(nums1, nums2, k) {
    let minheap = new MinPriorityQueue({priority: v => v[0]});
    minheap.enqueue([nums1[0]+nums2[0],0,0])
    let length1 = nums1.length
    let length2 = nums2.length
    let output = []
    while (minheap.size() && output.length < k){
        let [total, i, j] = minheap.dequeue().element
        output.push([nums1[i],nums2[j]])
        if (j + 1 < length2){
            minheap.enqueue([nums1[i]+nums2[j+1],i,j+1])}
        if (j === 0 && i+1 < length1){
            minheap.enqueue([nums1[i+1]+nums2[0],i+1,0])}}
    return output


};


var kSmallestPairs = function(nums1, nums2, k) {
    
    if (nums1.length === 0 || nums2.length === 0) return []
    
    let arr = [];
    let max = -Infinity;
    
    for (let i = 0; i < nums1.length; i++) {
        for (let j = 0; j < nums2.length; j++) {

            let obj = {
                sum: nums1[i] + nums2[j],
                nums: [nums1[i], nums2[j]]
            }
            
            if (obj.sum >= max && arr.length >= k) {
                break;
            } else if (obj.sum <= max && arr.length < k) {
                arr.push(obj);
            } else if (obj.sum > max && arr.length < k) {
                max = obj.sum; 
                arr.push(obj);
            } else if (obj.sum < max && arr.length >= k) {
                let newMax = -Infinity;
                let replaced = false;
                for (let n = 0; n < arr.length; n++) {
                    if (!replaced && arr[n].sum === max) {
                        arr[n] = obj;
                        replaced = true;
                    }
                    if (arr[n].sum > newMax) newMax = arr[n].sum
                }
                max = newMax;
            } 
        }
    }
    
    return arr.map(obj => obj.nums);
    
};


var MedianFinder = function() {
    this.lower = new MaxPriorityQueue()
    this.upper = new MinPriorityQueue()
    
};

MedianFinder.prototype.addNum = function(num) {

    this.lower.enqueue(num)

    this.upper.enqueue(this.lower.dequeue().element)
    
    if (this.lower.size() < this.upper.size()){
         this.lower.enqueue(this.upper.dequeue().element)}
    
};

MedianFinder.prototype.findMedian = function() {
    if (this.lower.size() > this.upper.size()){
        return this.lower.front().element}               
    else{
        return (this.upper.front().element + this.lower.front().element) / 2}
    
};

class MedianFinder {
    constructor() {
        this.lower = new MaxPriorityQueue(); 
        this.upper = new MinPriorityQueue(); 
    }

    addNum(num) {
        if (this.lower.size() === 0) {
            this.lower.enqueue(num);
        } else if (this.lower.size() !== this.upper.size()) {
            if (this.lower.front().element <= num) {
                this.upper.enqueue(num);
            } else {
                this.upper.enqueue(this.lower.dequeue().element);
                this.lower.enqueue(num);
            }
        } else {
            if (this.upper.front().element > num) {
                this.lower.enqueue(num);
            } else {
                this.lower.enqueue(this.upper.dequeue().element);
                this.upper.enqueue(num);
            }
        }
    }

    findMedian() {
        if (this.lower.size() === this.upper.size()) {
            return (this.lower.front().element + this.upper.front().element) / 2;
        } else {
            return this.lower.front().element;
        }
    }
}


var addBinary = function(a, b) {
    let i = a.length - 1, j = b.length - 1
    let output = ''
    let carry = 0
    while (i >= 0 || j >=0){
        let total = carry
        if (i>=0){
            total += parseInt(a[i])}
        if (j>= 0){
            total += parseInt(b[j])}
        carry = Math.floor(total / 2)
        output = (total%2).toString() + output
        i -= 1
        j -= 1}
    if (carry){
        return "1" + output}
    return output

    
};

var addBinary = function(a, b) {
    
    const aBin = `0b${a}`;
    const bBin = `0b${b}`;
  
    
    const sum = BigInt(aBin) + BigInt(bBin);
  
    
    return sum.toString(2);
  };

  
  let addBinary = (a, b) => {
    
    
    
    
    
    
  
    let carry = 0;
    let result = '';
  
    let len1 = a.length - 1;
    let len2 = b.length - 1;
  
    for (; len1 >= 0 || len2 >= 0 || carry > 0; len1--, len2--) {
      let sum = (+a[len1] || 0) + (+b[len2] || 0) + carry;
      if (sum > 1) {
        sum = sum % 2;
        carry = 1;
      } else {
        carry = 0;
      }
      result = `${sum}${result}`;
    }
    return result;
  };

  var addBinary = function(a, b) {
    return (BigInt("0b"+a) + BigInt("0b"+b)).toString(2);
}

var reverseBits = function(n) {
    let res = 0
    for(let i = 0; i < 32; i++){
        res = (res<<1) + (n&1)
        n>>=1}
    return res >>> 0
    
};

var reverseBits = function(n) {
    return BigInt("0b" + n.toString(2).padStart(32,'0').split('').reverse().join(''))
};

var reverseBits = function(n) {
    var result = 0;
    var count = 32;
  
    while (count--) {
      result *= 2;
      result += n & 1;
      n = n >> 1;
    }
    return result;
  };

function reverseBits(n) {
    return Number.parseInt(n.toString(2).split("").reverse().join("").padEnd(32, "0"), 2);
  }


var hammingWeight = function(n) {
    let count = 0
    while (n){
        n &= (n - 1)
        count += 1}
    return count

};

var hammingWeight = function(n) {
    return n.toString(2).split('').filter(bit => bit === '1').length;
};

var hammingWeight = function(n) {
    let res = 0
    while (n !== 0){
        res += (n&1)
        n>>>=1}
    return res
};

var hammingWeight = function(int) {
    const str = int.toString(2)
    return str === '0' ? 0 : (str.match(/1/g)).length; 
};

var hammingWeight = function(int) {
    return int.toString(2).replaceAll("0", "").length;    
};

var hammingWeight = function(int) {
    let count = 0;
    while (int !== 0) {
        const bitComparison = int & 1; 
        if (bitComparison === 1) count++;
        int >>>= 1; 
    }  
    return count;
};

var singleNumber = function(nums) {
    
    let uniqNum = 0;
    
    for (let idx = 0; idx < nums.length; idx++) {
        
        uniqNum = uniqNum ^ nums[idx];
    } return uniqNum;       
};

var singleNumber = function(nums) {
    return nums.reduce( (acc,num) => acc ^ num)
};


var singleNumber = function(nums) {
    let ans = 0

    for (let i = 0; i < 32; i++){
        let bit_sum = 0
        for (let num of nums){
            bit_sum += (num >> i) & 1}
        bit_sum %= 3
        ans |= bit_sum << i}

    return ans

};


var singleNumber = function(nums) {
    let ones = 0
    let twos = 0
    for (let num of nums){
        ones = (ones ^ num) & ~twos
        twos = (twos ^ num) & ~ones}
    return ones

};

function singleNumber(nums) {
    let ones = 0;
    let twos = 0;

    for (let num of nums) {
        twos = twos | (ones & num);
        ones = ones ^ num;
        let commonBitMask = ~(ones & twos);
        ones = ones & commonBitMask;
        twos = twos & commonBitMask;
    }

    return ones;
}


var rangeBitwiseAnd = function(left, right) {
    if ((left > 0 && right > 0) && Math.log2(right) - Math.log2(left) >= 1){
        return 0}
    let shift = 0
    while (left < right){
        left >>= 1
        right >>= 1
        shift += 1}
    return left << shift
    
};

var isPalindrome = function(x) {
    if (x < 0 || (x > 0 && x%10 == 0)){
        return false}
    
    let result = 0
    while (x > result){
        result = result * 10 + x % 10
        x = Math.floor(x / 10)}
        
    return (x == result || x == Math.floor(result / 10)) 

};

var isPalindrome = function(x) {
    if (x < 0) {
        return false;
    }
    
    return String(x) === String(x).split('').reverse().join('');
}

var isPalindrome = function(x) {
    var reverse = 0;
    var copy = x;

    
    
    while (copy > 0) {
      const digit = copy % 10;
      reverse = reverse * 10 + digit;
      copy = ~~(copy / 10);
    }

    return reverse == x;
};

function plusOne(digits) {
    for (let i = digits.length - 1; i >= 0; i--) {
      if (digits[i] !== 9) {
        digits[i]++;
        break;
      }
      digits[i] = 0;
    }
    if (digits[0] === 0) {
      return [1, ...digits];
    }
    return digits;
  }
  
  var plusOne = function(digits) {
    for(var i = digits.length - 1; i >= 0; i--){
         digits[i]++; 
        if(digits[i] > 9){
            digits[i] = 0;
        }else{
            return digits;
        }
    }
    digits.unshift(1);
    return digits;
    };

var plusOne = function(digits) {
    return (BigInt(digits.join("")) + BigInt(1)).toString().split("");
};

var plusOne = function(digits) {
    return Array.from(String(BigInt((digits.map(num => String(num))).join(''))+BigInt(1))).map(Number)

};


var trailingZeroes = function(n) {
    if (n < 5){
        return 0}
    else{
        return parseInt(n/5) + trailingZeroes(parseInt(n/5))}
    
};

var trailingZeroes = function(n) {
    let count = 0
    while (n > 0){
        n = Math.floor(n / 5)
        count += n}
    return count

    
};


var mySqrt = function(x) {
    let res = 1
    for (let i = 0; i < 20; i++){
        let temp = res
        res = temp - (temp**2 - x)/(2 * temp)}
    return Math.floor(res)
    
};

var mySqrt = function(x) {
    var left = 1;
    var right = Math.floor(x / 2) + 1;
    var mid;

    while (left <= right) {
        mid = Math.floor((left + right) / 2);

        if (mid * mid > x) {
            right = mid - 1;
        } else if (mid * mid < x) {
            left = mid + 1;
        } else {
            return mid;
        }
    }

    return right;
};

var myPow = function(x, n) {   
        
    function power_helper(x, n){
        let result = 1.0
        let current_product = x
        
        while (n > 0){
            if (n % 2 == 1){
                result *= current_product}
            current_product *= current_product
            n = Math.floor(n/2)}
        
        return result}
    
    if (n == 0){
        return 1.0}
    else if( n > 0){
        return power_helper(x, n)}
    else{
        return 1 / power_helper(x, -n)}
};

var myPow = function(x, n) {
    if (n===0) return 1;
    
    let pow = Math.abs(n);
    
	let result = pow%2===0 ? myPow(x*x,pow/2) : myPow(x*x,(pow-1)/2) * x;
    
    return n < 0 ? 1/result : result;
};

var myPow = function(a, b) {
    
    let flag = 0
	
	
    if(a<0) {
        a = Math.abs(a)
        if(b%2!=0) flag=1    
		let res =  Math.exp(  b*Math.log(a)  )
        return ( flag==1 ? res*-1 : res  )
    }
	
	
	
    
    else return Math.exp(  (b)  *  Math.log(a)  )
};


var maxPoints = function(points) {
    function gcd(a, b) {
        while (b !== 0) {
            [a, b] = [b, a % b];
        }
        return a;
    }

    points.sort((a, b) => {
        if (a[0] !== b[0]) {
            return a[0] - b[0];
        } else {
            return a[1] - b[1];
        }
    });

    let slope = new Map();
    let M = 0;
    for (let i = 0; i < points.length; i++) {
        slope.clear();
        let [x1, y1] = points[i];
        for (let [x2, y2] of points.slice(i + 1)) {
            let dx = x2 - x1;
            let dy = y2 - y1;
            let G = gcd(dx, dy);
            let m = `${dx/G},${dy/G}`;  
            
            if (slope.has(m)) {
                slope.set(m, slope.get(m) + 1);
            } else {
                slope.set(m, 1);
            }
            if (slope.get(m) > M) {
                M = slope.get(m);
            }
        }
    }
    return M + 1;
};


var maxPoints = function(points) {
    if (points.length <= 2) {
        return points.length;
    }
    
    function find_slope(p1, p2) {
        let [x1, y1] = p1;
        let [x2, y2] = p2;
        if (x1 - x2 === 0) {
            return Infinity;  
        }
        return (y1 - y2) / (x1 - x2);
    }
    
    let ans = 1;
    for (let i = 0; i < points.length; i++) {
        let slopes = new Map();
        let p1 = points[i];
        for (let j = i + 1; j < points.length; j++) {
            let p2 = points[j];
            let slope = find_slope(p1, p2);
            if (slopes.has(slope)) {
                slopes.set(slope, slopes.get(slope) + 1);
            } else {
                slopes.set(slope, 1);
            }
            ans = Math.max(slopes.get(slope), ans);
        }
    }
    return ans + 1;
};


var maxPoints = function(points) {
    let max = 0;
      
      for (const x of points) {
        const slopes = new Map();
      
      for (const y of points) {
        if (x === y) continue;
        let slope = Infinity;
        
        if (y[0] - x[0] !== 0) {
          
          slope = (y[1] - x[1]) / (y[0] - x[0]);
        }
        if (slopes.has(slope)) {
          slopes.set(slope, slopes.get(slope) + 1);
        } else {
          slopes.set(slope, 1);
        }
        max = Math.max(max, slopes.get(slope));
      }
    }
    return max + 1;
  };


var climbStairs = function(n) {
    if (n == 0 || n == 1){
        return 1}
    let prev = 1,curr = 1
    for (let i = 2; i<n+1; i++){
        let temp = curr
        curr = prev + curr
        prev = temp}
    return curr
};


var climbStairs = function(n) {
    if (n === 0 || n === 1) {
      return 1;
    }

    let dp = new Array(n + 1).fill(0);
    dp[0] = 1;
    dp[1] = 1;

    for (let i = 2; i <= n; i++) {
      dp[i] = dp[i - 1] + dp[i - 2];
    }

    return dp[n];

};

function climbStairs(n) {
    let a = 1, b = 1;
    
    for(let i = 1; i <= n; i++) {
      [a, b] = [b, a + b];
    }
    
    return a;
  }
  
var climbStairs = function(n, memo = new Array()) {
    if (n === 1) {
        return 1;
    }
    if (n === 2) {
        return 2;
    }
    if (memo[n] !== undefined) {
        return memo[n];
    }
    let res = climbStairs(n-1, memo) + climbStairs(n-2, memo);
    memo[n] = res;
    return res;}  


const mul = (
    [[a1, a2],[a3, a4]],
    [[b1, b2],[b3, b4]]) =>
    [[a1 * b1 + a2 * b3, a1 * b2 + a2 * b4],
        [a3 * b1 + a4 * b3, a3 * b2 + a4 * b4]];

const matrix = [[0, 1],[1, 1]];

const id = [[1, 0],[0, 1]]

var climbStairs = function(n) {
    let result = id;
    const bits = (n + 1).toString(2);

    for(const bit of bits){
        result = mul(result, result);
        if(bit === "1"){
            result = mul(result, matrix);
        }
    }
    return result[1][0];
}

var rob = function(nums) {
    let prev = 0
    let curr = 0
    for (let num of nums){
        [curr, prev] = [Math.max(curr,prev+num), curr]}
    return curr
    
};

function rob(nums) {
    if (!nums.length) {
      return 0;
    }
  
    if (nums.length === 1) {
      return nums[0];
    }
  
    nums.push(0);
    nums.reverse();
  
    
    for (let idx = 3; idx < nums.length; idx++) {
      const num = nums[idx];
      nums[idx] = Math.max(num + nums[idx - 2], num + nums[idx - 3]);
    }
  
    
    return Math.max(nums[nums.length - 1], nums[nums.length - 2]);
  }

  

var rob = function(nums) {
    return nums.reduce(function(p, n) { 
        return [p[1], Math.max(p[0] + n, p[1])]; 
    }, [0,0])[1];
};


var rob = function(nums) {
        
    if (!nums.length) return 0;
    if (nums.length === 1) return nums[0];
    if (nums.length === 2) return Math.max(nums[0], nums[1]);
    
    let maxAtTwoBefore = nums[0];
    let maxAtOneBefore = Math.max(nums[0], nums[1]);
    
    for (let i = 2; i < nums.length; i++) {
        const maxAtCurrent = Math.max(nums[i] + maxAtTwoBefore, maxAtOneBefore);
        
        maxAtTwoBefore = maxAtOneBefore;
        maxAtOneBefore = maxAtCurrent;
    }
    
    return maxAtOneBefore;
};


var wordBreak = function(s, wordDict) {
    wordDict = new Set(wordDict)
    let word_locations=[0]
    for(let i = 1 ; i < s.length+1;i++){
        for (let index of word_locations){
            if (wordDict.has(s.slice(index, i))){
                word_locations.push(i)
                break}}}
    return word_locations[word_locations.length-1] == s.length ? true: false
                    
    
};

function wordBreak(s, wordDict) {
    const memo = {};
    
    function construct(current) {
      if (current in memo) return memo[current];
      
      if (!current) return true;
      
      for (const word of wordDict) {
        if (current.startsWith(word)) {
          const newCurrent = current.slice(word.length);
          if (construct(newCurrent)) {
            memo[current] = true;
            return true;
          }
        }
      }
      
      memo[current] = false;
      return false;
    }
    
    return construct(s);
  }

  const wordBreak = (s, wordDict) => {
    const n = s.length;
    const dp = Array(n + 1).fill(false);
    dp[0] = true;
    
    const maxLen = Math.max(...wordDict.map(word => word.length)); 
    
    for (let i = 1; i <= n; i++) {
      for (let j = i - 1; j >= Math.max(i - maxLen - 1, 0); j--) { 
        if (dp[j] && wordDict.includes(s.substring(j, i))) {
          dp[i] = true;
          break;
        }
      }
    }
    
    return dp[n];
  };

  var wordBreak = function(s, wordDict) {
    let memo = {};
    let wordSet = new Set(wordDict);
    return dfs(s, wordSet, memo);
};

function dfs(s, wordSet, memo) {
    if (s in memo) return memo[s];
    if (wordSet.has(s)) return true;
    for (let i = 1; i < s.length; i++) {
        let prefix = s.substring(0, i);
        if (wordSet.has(prefix) && dfs(s.substring(i), wordSet, memo)) {
            memo[s] = true;
            return true;
        }
    }
    memo[s] = false;
    return false;
}


const wordBreak = (s, wordDict) => {
    if (wordDict == null || wordDict.length === 0) return false;
    const set = new Set(wordDict);
  
    
    
    
    const visited = new Set();
    const q = [0];
  
    while (q.length) {
      const start = q.shift();
  
      if (!visited.has(start)) {
        for (let end = start + 1; end <= s.length; end++) {
          if (set.has(s.slice(start, end))) {
            if (end === s.length) return true;
            q.push(end);
          }
        }
        visited.add(start);
      }
    }
    return false;
  };


var coinChange = function(coins, amount) {
    dp = Array.from({ length: amount + 1 }, () => amount + 1)
    dp[0] = 0

    for (let coin of coins){
      for(let i = coin; i < amount + 1; i++){
        dp[i] = Math.min(dp[i], dp[i - coin] + 1)}}

    return  dp[amount] == amount + 1 ? -1 : dp[amount]
    
};


function coinChange(coins, amount) {
    function coinChangeInner(rem, cache) {
        if (rem < 0) {
            return Infinity;
        }
        if (rem === 0) {
            return 0;
        }
        if (cache[rem] !== undefined) {
            return cache[rem];
        }
        
        let minCoins = Infinity;
        for (let coin of coins) {
            let res = coinChangeInner(rem - coin, cache) + 1;
            minCoins = Math.min(minCoins, res);
        }
        
        cache[rem] = minCoins;
        return cache[rem];
    }
    
    const ans = coinChangeInner(amount, {});
    return ans === Infinity ? -1 : ans;
}

var coinChange = function(coins, amount) {
    let dp = Array(amount + 1).fill(Infinity)
    dp[0] = 0

    for (let coin of coins){
      for(let i = coin; i < amount + 1; i++){
        dp[i] = Math.min(dp[i], dp[i - coin] + 1)}}

    return  dp[amount] == Infinity ? -1 : dp[amount]
    
};

var lengthOfLIS = function(nums) {
    let res = []

    for (let num of nums){
        if (res.length == 0 || num > res[res.length - 1]){
            res.push(num)}
        else{
            let low = 0, high = res.length
            while( low < high){
                let mid = Math.floor((low + high)/2)
                if (num > res[mid]){
                    low = mid + 1}
                else{
                    high = mid}}
            res[low] = num}}

    return res.length
    
};


var lengthOfLIS = function(nums) {
    const dp = new Array(nums.length).fill(1);
    let output = 1;
  
    for (let i = 0; i < nums.length; i++) {
      for (let j = i - 1; j >= 0; j--) {
        if (nums[i] > nums[j]) {
          dp[i] = Math.max(dp[j] + 1, dp[i]);
          output = Math.max(output, dp[i]);
        }
      }
    }
  
    return output;
  }

  var minimumTotal = function(triangle) {
    for( let i = triangle.length - 2 ; i > -1 ;i--){
        for (let j = 0; j < i+1 ;j++){                
            triangle[i][j] += Math.min(triangle[i+1][j],    
                                    triangle[i+1][j+1]) }}
    return triangle[0][0]

    
};

var minimumTotal = function(triangle) {
    let dp = triangle[triangle.length - 1];
    for (let i = triangle.length - 2; i >= 0; i--) {
        for (let j = 0; j < triangle[i].length; j++) {
            dp[j] = triangle[i][j] + Math.min(dp[j], dp[j + 1]);
        }
    }
    return dp[0];
}

var minimumTotal = function(triangle) {
    for (let i = 1; i < triangle.length; i++) {
        for (let j = 0; j <= i; j++) {
            triangle[i][j] += Math.min(
                triangle[i - 1][j - (j === i ? 1 : 0)], 
                triangle[i - 1][j - (j > 0 ? 1 : 0)]   
            );
        }
    }
    return Math.min(...triangle[triangle.length - 1]);
}

var minimumTotal = function(triangle) {
    function dfs(level, i) {
        if (level >= triangle.length) return 0;
        return triangle[level][i] + Math.min(dfs(level + 1, i), dfs(level + 1, i + 1));
    }
    return dfs(0, 0);
}
var minPathSum = function(grid) {
            
    let m = grid.length, n = grid[0].length
    
    for (let i = 1; i < m; i++){
        grid[i][0] += grid[i-1][0]}
    
    for (let i = 1; i < n; i++){
        grid[0][i] += grid[0][i-1]}
    
    for (let i = 1; i < m; i++){
        for (let j = 1; j < n; j++){
            grid[i][j] += Math.min(grid[i-1][j], grid[i][j-1])}}
    
    return grid[grid.length-1][grid[0].length -1]

    
};


var minPathSum = function(grid) {
    for (let i = 0; i < grid.length; i++) {
        for (let j = 0; j < grid[i].length; j++) {
            if (i === 0 && j === 0) {
                
                continue;
            }
            let above = i > 0 ? grid[i - 1][j] : Number.POSITIVE_INFINITY;
            let left = j > 0 ? grid[i][j - 1] : Number.POSITIVE_INFINITY;
            grid[i][j] += Math.min(above, left);
        }
    }
    return grid[grid.length - 1][grid[0].length - 1];
}

var uniquePathsWithObstacles = function(obstacleGrid) {
    if (obstacleGrid.length == 0 || obstacleGrid[0].length == 0 || obstacleGrid[0][0] == 1){
        return 0}
    for (let i = 0; i < obstacleGrid.length ; i++){
        for (let j = 0; j < obstacleGrid[i].length ; j++){
            if (obstacleGrid[i][j] == 1 || (i == 0 && j ==0)){
                obstacleGrid[i][j] ^= 1
                continue}
            let above = (i-1) < 0 ?  0 : obstacleGrid[i-1][j]
            let left = (j-1) < 0 ?  0 : obstacleGrid[i][j-1]
            obstacleGrid[i][j] += above + left}}
    return obstacleGrid[obstacleGrid.length-1][obstacleGrid[0].length-1]
    
};



function uniquePathsWithObstacles(obstacleGrid) {
    if (!obstacleGrid || !obstacleGrid.length || !obstacleGrid[0].length || obstacleGrid[0][0] === 1) {
        return 0;
    }

    let m = obstacleGrid.length;
    let n = obstacleGrid[0].length;

    let previous = new Array(n).fill(0);
    let current = new Array(n).fill(0);
    previous[0] = 1;

    for (let i = 0; i < m; i++) {
        current[0] = obstacleGrid[i][0] === 1 ? 0 : previous[0];
        for (let j = 1; j < n; j++) {
            current[j] = obstacleGrid[i][j] === 1 ? 0 : current[j - 1] + previous[j];
        }
        previous = [...current]; 
    }

    return previous[n - 1];
}

function uniquePathsWithObstacles(obstacleGrid) {
    const M = obstacleGrid.length;
    const N = obstacleGrid[0].length;
    const cache = {}; 
  
    const dfs = (i, j) => {
      const key = `${i},${j}`; 
      if (key in cache) {
        return cache[key]; 
      }
  
      if (obstacleGrid[i][j]) {
        return 0; 
      }
      if (i === M - 1 && j === N - 1) {
        return 1; 
      }
  
      let count = 0;
      if (i < M - 1) {
        count += dfs(i + 1, j); 
      }
      if (j < N - 1) {
        count += dfs(i, j + 1); 
      }
  
      cache[key] = count; 
      return count;
    };
  
    return dfs(0, 0);
  }

var longestPalindrome = function(s) {
    if (s.length <= 1){
        return s}

    function expand_from_center(left, right){
        while (left >= 0 && right < s.length && s[left] == s[right]){
            left -= 1
            right += 1}
        return s.slice(left+1,right)}

    let max_str = s[0]

    for (let i = 0; i < s.length - 1; i++){
        let odd = expand_from_center(i, i)
        let even = expand_from_center(i, i + 1)

        if (odd.length > max_str.length){
            max_str = odd}
        if (even.length > max_str.length){
            max_str = even}}

    return max_str
};


function longestPalindrome(s) {
    if (s.length <= 1) {
        return s;
    }

    let maxLen = 1;
    let maxStr = s[0];
    s = '#' + s.split('').join('#') + '#';
    let dp = new Array(s.length).fill(0);
    let center = 0;
    let right = 0;

    for (let i = 0; i < s.length; i++) {
        if (i < right) {
            dp[i] = Math.min(right - i, dp[2 * center - i]);
        }
        while (i - dp[i] - 1 >= 0 && i + dp[i] + 1 < s.length && s[i - dp[i] - 1] === s[i + dp[i] + 1]) {
            dp[i]++;
        }
        if (i + dp[i] > right) {
            center = i;
            right = i + dp[i];
        }
        if (dp[i] > maxLen) {
            maxLen = dp[i];
            maxStr = s.substring(i - dp[i], i + dp[i] + 1).replace(/#/g, '');
        }
    }

    return maxStr;
}

function longestPalindrome(s) {
    let output1 = 0;
    let output2 = 0;
    const stringLength = s.length;
    const dp = Array.from({ length: stringLength }, () => new Array(stringLength).fill(0));

    for (let i = 0; i < stringLength; i++) {
        dp[i][i] = 1;
        if (i + 1 < stringLength && s[i] === s[i + 1]) {
            dp[i][i + 1] = 1;
            output1 = i;
            output2 = i + 1;
        }
    }

    for (let i = 2; i < stringLength; i++) {
        for (let j = 0; j < stringLength - i; j++) {
            if (s[j] === s[j + i] && dp[j + 1][j + i - 1]) {
                dp[j][j + i] = 1;
                output1 = j;
                output2 = j + i;
            }
        }
    }

    return s.substring(output1, output2 + 1);
}



var longestPalindrome = function(s) {
    let longest = '';
    const findLongestPalindrome = (str, i, j) => {
        while(i >= 0 && j < str.length && str[i] === str[j]) {
            i -= 1;
            j += 1;
        }
        
        return str.slice(i + 1, j);
    }
    for (let i = 0; i < s.length; i++) {
        
        const current1 = findLongestPalindrome(s, i, i);
        const current2 = findLongestPalindrome(s, i, i + 1);
        const longerPalindrome = 
              current1.length > current2.length ? current1 : current2;
        if (longerPalindrome.length > longest.length) {
            longest = longerPalindrome;
        } 
    }
    return longest;
};


var isInterleave = function(s1, s2, s3) {
    let m = s1.length,  n = s2.length  ,l = s3.length
    if (m + n != l){
        return false}
    
    let memo = new Map() 
    
    function helper(i, j, k){
        if (k == l){
            return true}
        let key = `${i},${j}`
        if (memo.has(key)){

            return memo.get(key)}
        
        let ans = false
        if (i < m && s1[i] == s3[k]){
            ans = ans || helper(i + 1, j, k + 1)}
            
        if (j < n && s2[j] == s3[k]){
            ans = ans || helper(i, j + 1, k + 1)}
        
        memo.set(key,ans)
        return ans}
    
    return helper(0, 0, 0)
    
};


var isInterleave = function(s1, s2, s3) {
    let m = s1.length, n = s2.length, l = s3.length;
    if (m + n !== l) return false;

    let dp = new Array(n + 1).fill(false);
    dp[0] = true;

    for (let j = 1; j <= n; ++j) {
        dp[j] = dp[j - 1] && s2[j - 1] === s3[j - 1];
    }

    for (let i = 1; i <= m; ++i) {
        dp[0] = dp[0] && s1[i - 1] === s3[i - 1];
        for (let j = 1; j <= n; ++j) {
            dp[j] = (dp[j] && s1[i - 1] === s3[i + j - 1]) || (dp[j - 1] && s2[j - 1] === s3[i + j - 1]);
        }
    }
    
    return dp[n];
};


function isInterleave(s1, s2, s3) {
    let m = s1.length, n = s2.length, l = s3.length;
    if (m + n !== l) {
        return false;
    }

    let dp = Array.from({ length: m + 1 }, () => Array(n + 1).fill(false));
    dp[0][0] = true;

    for (let i = 1; i <= m; i++) {
        dp[i][0] = dp[i - 1][0] && s1[i - 1] === s3[i - 1];
    }

    for (let j = 1; j <= n; j++) {
        dp[0][j] = dp[0][j - 1] && s2[j - 1] === s3[j - 1];
    }

    for (let i = 1; i <= m; i++) {
        for (let j = 1; j <= n; j++) {
            dp[i][j] = (dp[i - 1][j] && s1[i - 1] === s3[i + j - 1]) || (dp[i][j - 1] && s2[j - 1] === s3[i + j - 1]);
        }
    }

    return dp[m][n];
}

var minDistance = function(word1, word2) {
    let m = word1.length, n = word2.length
    
    let dp = new Array(n + 1).fill(0);
    
    for (let j = 1; j < n+1; j++){
        dp[j] = dp[j-1] + 1}
    
    for (let i = 1; i < m+1; i++){
        let previous_diag = dp[0]
        dp[0] = dp[0] + 1
        for (let j = 1; j < n+1; j++){
            if( word1[i - 1] == word2[j - 1]){
                [dp[j], previous_diag] = [previous_diag, dp[j]]}
            else{
                [dp[j], previous_diag] = [1 + Math.min(
                    dp[j],    
                    dp[j - 1],    
                    previous_diag 
            ), dp[j]]}}}

    return dp[n]
    
};

function minDistance(word1, word2) {
    let m = word1.length, n = word2.length;
    
    
    let dp = Array.from({ length: m + 1 }, () => Array(n + 1).fill(0));
    
    
    for (let j = 1; j <= n; j++) {
        dp[0][j] = j;  
    }
    
    
    for (let i = 1; i <= m; i++) {
        dp[i][0] = i;  
    }
    
    
    for (let i = 1; i <= m; i++) {
        for (let j = 1; j <= n; j++) {
            if (word1[i - 1] === word2[j - 1]) {
                dp[i][j] = dp[i - 1][j - 1];  
            } else {
                dp[i][j] = 1 + Math.min(
                    dp[i - 1][j],    
                    dp[i][j - 1],    
                    dp[i - 1][j - 1] 
                );
            }
        }
    }
    
    
    return dp[m][n];
}


function maxProfit(prices) {
    let dp_2_hold = -Infinity, dp_2_not_hold = 0;
    let dp_1_hold = -Infinity, dp_1_not_hold = 0;
    
    for (let stock_price of prices) {
        
        dp_2_not_hold = Math.max(dp_2_not_hold, dp_2_hold + stock_price);
        dp_2_hold = Math.max(dp_2_hold, dp_1_not_hold - stock_price);
        
        
        dp_1_not_hold = Math.max(dp_1_not_hold, dp_1_hold + stock_price);
        dp_1_hold = Math.max(dp_1_hold, -stock_price);
    }
    
    return dp_2_not_hold;
}

var maxProfit = function(prices) {
    let buy1 = Infinity, buy2 = Infinity
    let sell1 = 0, sell2 = 0

    for(let price of prices){
        buy1 = Math.min(buy1, price)
        sell1 = Math.max(sell1, price - buy1)
        buy2 = Math.min(buy2, price - sell1)
        sell2 = Math.max(sell2, price - buy2)}

    return sell2
    
};
[1,3,4]


function maxProfit(prices) {
    if (!prices || prices.length === 0) {
      return 0;
    }

    
    const dp = [];
    for (let i = 0; i < prices.length; i++) {
      dp[i] = [];
      for (let j = 0; j < 3; j++) {
        dp[i][j] = [0, 0];
      }
    }

    
    dp[0][0][0] = 0;
    dp[0][0][1] = -prices[0];
    dp[0][1][0] = 0;
    dp[0][1][1] = -prices[0];
    dp[0][2][0] = 0;
    dp[0][2][1] = -prices[0];

    for (let i = 1; i < prices.length; i++) {
      for (let j = 1; j < 3; j++) {
        
        dp[i][j][0] = Math.max(dp[i - 1][j][0], dp[i - 1][j][1] + prices[i]);
        
        dp[i][j][1] = Math.max(dp[i - 1][j][1], dp[i - 1][j - 1][0] - prices[i]);
      }
    }

    
    return dp[prices.length - 1][2][0]
  }


  var maxProfit = function(k, prices) {
    if (k >= Math.floor(prices.length/2)){
       return prices
  .map((price, i, arr) => i > 0 ? Math.max(0, price - arr[i - 1]) : 0)
  .reduce((acc, profit) => acc + profit, 0);}

    let ans = Array(prices.length).fill(0)
    for(let _ = 0; _ < k; _++){
        let most = 0
        for(let i = 1; i < prices.length; i++){
            most = Math.max(ans[i], most + prices[i] - prices[i-1])
            ans[i] = Math.max(ans[i-1], most)}}
    return ans[ans.length-1]
    
};


var maxProfit = function(k, prices) {
    if (k >= prices.length / 2) {
          let sum = 0;
          for (let i = 1; i < prices.length; i++) {
            sum += Math.max(0, prices[i] - prices[i - 1]);
          }
          return sum;
        }
    
        let buy = new Array(k).fill(Infinity);
        let sell = new Array(k).fill(0);
    
        for (let price of prices) {
          for (let i = 0; i < k; i++) {
            buy[i] = i ? Math.min(buy[i], price - sell[i - 1]) : Math.min(buy[i], price);
            sell[i] = Math.max(sell[i], price - buy[i]);
          }
        }
    
        return k && prices.length ? sell[k - 1] : 0;
      }

      
      function maxProfit(k, prices) {
        if (!prices || prices.length === 0) {
          return 0;
        }
      
        
        
        let dp = Array.from({ length: prices.length }, () =>
          Array.from({ length: k + 1 }, () => Array(2).fill(0))
        );
      
        for (let i = 0; i <= k; i++) {
          dp[0][i][1] = -prices[0];
        }
      
        for (let i = 1; i < prices.length; i++) {
          for (let j = 1; j <= k; j++) {
            
            dp[i][j][0] = Math.max(dp[i - 1][j][0], dp[i - 1][j][1] + prices[i]);
            
            dp[i][j][1] = Math.max(dp[i - 1][j][1], dp[i - 1][j - 1][0] - prices[i]);
          }
        }
      
        
        return dp[prices.length - 1][k][0];
      }

      class MinHeap {
        constructor() {
            this.heap = [];
        }
    
        getParentIndex(i) {
            return Math.floor((i - 1) / 2);
        }
    
        getLeftChildIndex(i) {
            return 2 * i + 1;
        }
    
        getRightChildIndex(i) {
            return 2 * i + 2;
        }
    
        swap(i1, i2) {
            [this.heap[i1], this.heap[i2]] = [this.heap[i2], this.heap[i1]];
        }
    
        push(element) {
            this.heap.push(element);
            let index = this.heap.length - 1;
            let parent = this.getParentIndex(index);
            while (index > 0 && this.heap[parent][0] > this.heap[index][0]) {
                this.swap(parent, index);
                index = parent;
                parent = this.getParentIndex(index);
            }
        }
    
        pop() {
            const item = this.heap[0];
            this.heap[0] = this.heap[this.heap.length - 1];
            this.heap.pop();
            this.heapify(0);
            return item;
        }
    
        heapify(index) {
            let left = this.getLeftChildIndex(index);
            let right = this.getRightChildIndex(index);
            let smallest = index;
    
            if (left < this.heap.length && this.heap[left][0] < this.heap[smallest][0]) {
                smallest = left;
            }
    
            if (right < this.heap.length && this.heap[right][0] < this.heap[smallest][0]) {
                smallest = right;
            }
    
            if (smallest !== index) {
                this.swap(index, smallest);
                this.heapify(smallest);
            }
        }
    
        isEmpty() {
            return this.heap.length === 0;
        }
    }
    
    class DoubleLinkListNode {
        constructor(ind, pre = null, next = null) {
            this.ind = ind;  
            this.pre = pre || this;  
            this.next = next || this;  
        }
    }
    
    
    function minMaxList(arr) {
        let n = arr.length;  
        if (n === 0) {  
            return [];
        }
        let sign = -1;  
        let res = [9999];  
        arr.forEach(num => {
            
            if (num * sign > res[res.length - 1] * sign) {
                res[res.length - 1] = num;
            } else {
                
                res.push(num);
                sign *= -1;
            }
        });
        
        if (res.length % 2 === 1) {
            res.pop();
        }
        return res;
    }
    
    
    function maxProfit(k, prices) {
        let newP = minMaxList(prices);  
        let n = newP.length;  
        let m = Math.floor(n / 2);  
        let res = 0;  
        
        for (let i = 0; i < m; i++) {
            res += newP[i * 2 + 1] - newP[i * 2];
        }
        
        if (m <= k) {
            return res;
        }
    
        
        let head = new DoubleLinkListNode(-1);
        let tail = new DoubleLinkListNode(-1);
        let NodeList = [new DoubleLinkListNode(0, head)];
        for (let i = 1; i < n; i++) {
            NodeList.push(new DoubleLinkListNode(i, NodeList[i - 1]));
            NodeList[i - 1].next = NodeList[i];
        }
        NodeList[n - 1].next = tail;
        head.next = NodeList[0];
        tail.pre = NodeList[n - 1];
    
        
        let heap = new MinHeap();
        for (let i = 0; i < n - 1; i++) {
            if (i % 2 === 1) {
                heap.push([newP[i] - newP[i + 1], i, i + 1, 0]);
            } else {
                heap.push([newP[i + 1] - newP[i], i, i + 1, 1]);
            }
        }
    
        
        while (m > k) {
            let [loss, i, j, t] = heap.pop();
            if (!NodeList[i] || !NodeList[j]) {
                continue;
            }
            m -= 1;
            res -= loss;
            let nodei = NodeList[i], nodej = NodeList[j];
            let nodel = nodei.pre, noder = nodej.next;
            let l = nodel.ind, r = noder.ind;
            let valL = newP[l], valR = newP[r];
            noder.pre = nodel;
            nodel.next = noder;
            NodeList[i] = null;
            NodeList[j] = null;
            if (t === 0) {
                heap.push([valR - valL, l, r, 1]);
            } else if (l !== -1 && r !== -1) {
                heap.push([valL - valR, l, r, 0]);
            }
    
        }
    
        return res;
    }
    var maximalSquare = function(matrix) {
    let m= matrix.length, n = matrix[0].length
    
    let dp = Array(n).fill(0)
    let largest = 0

    for (let j = 0; j < n; j++){
        dp[j] = parseInt(matrix[0][j])
        largest = Math.max(largest,dp[j])}
    
    for (let i = 1; i < m; i++){
        let previous_diag = dp[0]
        dp[0] = parseInt(matrix[i][0])
        largest = Math.max(largest,dp[0])
        for (let j = 1; j < n; j++){
            if (matrix[i][j] == '1'){
                [dp[j], previous_diag] = [(Math.min(dp[j-1],dp[j],previous_diag) + 1), dp[j]]
                largest = Math.max(largest,dp[j])}
            else{
                [dp[j], previous_diag] = [0, dp[j]]}}}
            
    return largest*largest
        
    
};


var maximalSquare = function (matrix) {
    let max = 0
    for (let i = 0; i < matrix.length; i++) {
      for (let j = 0; j < matrix[0].length; j++) {
        if (matrix[i][j] === "0") continue
        if(i > 0 && j > 0)
          matrix[i][j] = Math.min(matrix[i - 1][j], matrix[i][j - 1], matrix[i - 1][j - 1]) + 1
        max = Math.max(matrix[i][j], max)
      }
    }
    return max ** 2
  }


  var maximalSquare = function(matrix) {
    let m= matrix.length, n = matrix[0].length
    
    let dp = Array(n).fill(0)
    let largest = 0

    for (let j = 0; j < n; j++){
        dp[j] = matrix[0][j]
        largest = Math.max(largest,dp[j])}
    
    for (let i = 1; i < m; i++){
        let previous_diag = dp[0]
        dp[0] = matrix[i][0]
        largest = Math.max(largest,dp[0])
        for (let j = 1; j < n; j++){
            if (matrix[i][j] == '1'){
                [dp[j], previous_diag] = [(Math.min(dp[j-1],dp[j],previous_diag) + 1), dp[j]]
                largest = Math.max(largest,dp[j])}
            else{
                [dp[j], previous_diag] = [0, dp[j]]}}}
            
    return largest*largest
        
    
};


function maximalSquare(matrix) {
    let m = matrix.length;
    let n = matrix[0].length;
    
    
    let dp = new Array(m).fill(0).map(() => new Array(n).fill(0));
    let largest = 0;

    
    for (let j = 0; j < n; j++) {
        dp[0][j] = parseInt(matrix[0][j]);
        largest = Math.max(largest, dp[0][j]);
    }
    
    
    for (let i = 0; i < m; i++) {
        dp[i][0] = parseInt(matrix[i][0]);
        largest = Math.max(largest, dp[i][0]);
    }

    
    for (let i = 1; i < m; i++) {
        for (let j = 1; j < n; j++) {
            if (matrix[i][j] === '1') {
                
                dp[i][j] = Math.min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1;
                largest = Math.max(largest, dp[i][j]);
            }
        }
    }

    
    return largest * largest;
}
