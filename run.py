from main import predictions, testX
print("====CAPTCHA THE FLAG====")
#to be altered to run later with pre-trained model
index = input("Insert a number to represent an index in the test array: ")

predictions(testX[int(index)])