import warnings
warnings.simplefilter('ignore', UserWarning)

from joblib import load

# Load the trained model
def load_model(model_file):
    try: 
        rfc = load(model_file)
        return rfc
    except Exception as e:
        print("Error loading the model:", e)
        return None



# Function to get user inputs
def get_user_inputs():
    no_of_dependents = int(input("How many dependents do you have: "))
    education = input("What is your college education level ('Graduate' or 'Not Graduate'): ").lower()
    self_employed = input("Are you self-employed? (yes/no): ").lower()
    income_annum = float(input("Annual income: "))
    loan_amount = float(input("Loan amount: "))
    loan_term = int(input("Loan term (in years): "))
    cibil_score = int(input("What is your CIBIL score: "))
    residential_assets_value = float(input("Value of residential assets: "))
    commercial_assets_value = float(input("Value of commercial assets: "))
    luxury_assets_value = float(input("Value of luxury assets: "))
    bank_asset_value = float(input("Value of bank assets: "))
    
    return [no_of_dependents, education, self_employed, income_annum, 
            loan_amount, loan_term, cibil_score, residential_assets_value, 
            commercial_assets_value, luxury_assets_value, bank_asset_value]



# Function to preprocess user inputs
def preprocess_inputs(inputs):
    # Convert education to numerical values
    education_mapping = {'graduate': 0, 'not graduate': 1}
    inputs[1] = education_mapping.get(inputs[1], 0)
    
    # Convert self_employed to binary
    inputs[2] = 0 if inputs[2] == 'yes' else 1
    
    return inputs


# Function to make prediction
def predict_approval(model, inputs):
    # Get probability estimates for both classes
    proba = model.predict_proba([inputs])[0]
    
    # Probability of class 0 (approval)
    approval_proba = proba[0]
    
    # Return probability of approval
    return approval_proba



def display_outcome(prediction):
    print(f"Probability of approval: {prediction*100}%")
    if prediction > 0.5:
        print('Application outcome prediction: Approved')
    else:
        print('Application outcome prediction: Rejected')



# Main function
def main():
    while True:
        print("*** WELCOME TO THE LOAN APPROVAL INDICATOR ***")
        print("Please input the following information.")
        
        # Load the trained model
        model_file = 'trained_rfc.joblib'
        model = load_model(model_file)
        
        # Get user inputs
        user_inputs = get_user_inputs()

        # Preprocess user inputs
        preprocessed_inputs = preprocess_inputs(user_inputs)

        # Make prediction
        prediction = predict_approval(model, preprocessed_inputs)

        # Display prediction
        #print("Probability of approval: ", prediction)
        display_outcome(prediction)
        
        # Ask the user if they want to continue or quit
        choice = input("Do you want to start over? (y/n): ").lower()
        if choice != 'y':
            print("Goodbye!")
            break


if __name__ == "__main__":
    main()