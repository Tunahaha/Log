def multiplication_or_sum(num1, num2):
  product = num1 *num2
  if(product < -1000):
    return product
  else:
    return num1 +num2

if __name__ == '__main__':   # pragma: no cover
  print(multiplication_or_sum(10, 20))
