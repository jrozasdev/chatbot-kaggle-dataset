"""
Small chat application to communicate with the virtual assistant chatbot.
"""

from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle

# Load the tokenizer from disk
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Load the label encoder from disk
with open('encoder.pkl', 'rb') as handle:
    label_encoder = pickle.load(handle)

# Importing the trained saved model
model = keras.models.load_model('chatbot_trained_model')

max_length = 100

# Responses dictionary to assign a response to each intent
responses = {
    'cancel_order': 'You can cancel your order by navigating to your order status from the Orders tab and selecting Cancel.',
    'complaint': 'You can file a complaint by sending an email to complaints@business.website.',
    'contact_customer_service': 'The customer service contact info is in the Contact tab.',
    'contact_human_agent': 'To contact a human agent you can send an email or call the number in the Contact tab.',
    'create_account': 'You can create an account by following the steps after clicking "sign up".',
    'change_order': 'To change your order click "modify order" in the Orders tab.',
    'change_shipping_address': 'You can change your shipping address by clicking "edit profile" in your profile page.',
    'check_cancellation_fee':'You can check the cancellation fee of your order in the order tab',
    'check_invoices': 'You can check invoices in your order history tab',
    'check_payment_methods': 'You can check payment methods in your profile or when finishing up an order',
    'check_refund_policy': 'The refund policy can be fount in your order page.',
    'delete_account': 'You can delete your account by clicking "delete account" in your profile.',
    'delivery_options': 'You can check delivery options when finishin up an order.',
    'delivery_period': 'The delivery period appears on the order.',
    'edit_account': 'You can edit your account in the profile page.',
    'get_invoice': 'You can get an invoice by requesting it from the order page.',
    'get_refund': 'To get a refund, go to your order and request it by clicking "refund product" and following the instructions.',
    'newsletter_subscription': 'You can subscribe to the newsletter by clicking on "subscribe to newsletter".',
    'payment_issue': 'To solve an issue with payment, contact customer support.',
    'place_order': 'To place an order you can click on "place order" from a product page.',
    'recover_password': 'To recover your password click on "recover password" from the log in page.',
    'registration_problems': 'If you encounter any trouble registering, contact our customer support via email or phone',
    'review': 'You can write a review in the product page by clicking the "add review" button.',
    'set_up_shipping_address': 'You can set up a shipping address in the order page or save one in your profile',
    'switch_account': 'To switch account, log out and log in to your desired account',
    'track_order': 'From the order tab you can track the order status',
    'track_refund': 'Your refund status should appear in the order history'
}

# Chat function
while True:

    print("You:")
    text = input()

    result = ''

    if text == 'quit':
        break

    input_data = pad_sequences(tokenizer.texts_to_sequences([text]), truncating='post', maxlen=max_length)
    prediction = model.predict(input_data, verbose=0)
    tag = label_encoder.inverse_transform([np.argmax(prediction)])

    print(responses[tag[0]])

    print()