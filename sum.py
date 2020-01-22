# this method takes an array, its length and a requested sum and returns three
# numbers from the array whose sum is equal to said requested sum. If no such
# numbers exist, return -1,
def find_num(array, len, sum):
    array.sort()
    for i in range(0, len - 2):
        l = i + 1 # starting index of the second item
        r = len - 1 # starting index of the third item

        while(l != r):
            if(array[i] + array[l] + array[r] > sum):
                r -= 1 # if the current sum is too big, choose a smaller third item
            elif(array[i] + array[l] + array[r] < sum):
                l += 1 # if the current sum is too low, choose a bigger second item
            else:
                return array[i], array[l], array[r] # return the correct items

    return -1, -1, -1 # if you got here, you didn't find a correct sequence


a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
print(find_num(a, 10, 7))