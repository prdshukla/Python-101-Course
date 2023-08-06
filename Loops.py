def to_celcius(temp):
    return (temp-32)*5/9

# function implemetation
print (to_celcius(453))

# string manipulation
for left in range(9):
    for right in range(left,9):
        print("["+str(right) + "|" +str(left) + "]", end=" ")
    print()