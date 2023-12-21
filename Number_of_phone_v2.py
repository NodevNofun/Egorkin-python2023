import re


def normalize_phone_number(number):
    number = re.sub(r'\D', '', number)  # Remove all non-digit characters
    return '+7' + number[-10:] if len(number) == 11 and (
            number.startswith('7') or number.startswith('8')) else '+7495' + number[-7:]


phone_number = ['+7-4-9-5-43-023-97', '4-3-0-2-3-9-7', '8-495-430']

new_phone_number = str(input())
normalize_new_phone_number = normalize_phone_number(new_phone_number)

phone_number_normal = list(map(normalize_phone_number, phone_number))
print(phone_number_normal)

massive_yes_no = ['YES' if number == normalize_new_phone_number else 'NO' for number in phone_number_normal]
print(massive_yes_no)
