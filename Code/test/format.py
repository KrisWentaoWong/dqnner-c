a = 12.123456789012345678901234567890
print format(a, "0.1")
print format(a, "0.5")
print format(a, "0.10")
print format(a, "0.10f")
print format(a, "0.20f")

print "bbbbbbbbbbbbbbbbbbbbbbbb"
b = 0.000025
for i in range(0, 10):
    print format(b, "0.10f") + "\t" + str(b)
    b = b/10

print "bbbbbbbbbbbbbbbbbbbbbbbb"
b = 0.000025
for i in range(0, 10):
    print format(b, "0.10f") + "\t" + str(b)
    b = b/2

