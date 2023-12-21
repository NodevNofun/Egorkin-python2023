def normalize(number):
    number_normalized = number.replace('(', "").replace(')', "").replace('-', "")
    if number_normalized[0] == '+' and len(number_normalized) == 12:
        return number_normalized
    elif number_normalized[0] == '8' and len(number_normalized) == 11:
        number_normalized.replace('8', '+7')
        return number_normalized
    elif len(number_normalized) == 7:
        number_normalized = '+7495' + number_normalized
        return number_normalized


new_number = str(input())
new_number = normalize(new_number)
phone_number = ['+7-4-9-5-43-023-97', '4-3-0-2-3-9-7', '8-495-430']
for i in range(len(phone_number)):
    phone_number[i] = normalize(phone_number[i])

for i in range(len(phone_number)):
    if new_number == phone_number[i]:
        print('YES')
    else:
        print('NO')
