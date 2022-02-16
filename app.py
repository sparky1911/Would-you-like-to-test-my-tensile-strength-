import streamlit as st
import pandas as pd
import shap
import pickle


st.write("""
# TENSILE STRENGTH PREDICTION FOR DIFFERENT STEEL ALLOYS

""")
st.write('---')

# Loads the Boston House Price Dataset


# Sidebar
# Header of Specify Input Parameters
st.sidebar.header('Specify Input Parameters')

def user_input_features():
    CARBON = st.number_input('CARBON',min_value=None, max_value=None)
    MANGANESE = st.number_input('MANGANESE',min_value=None, max_value=None)
    SILICON = st.number_input('SILICON',min_value=None, max_value=None)
    CHROMIUM = st.number_input('CHROMIUM',min_value=None, max_value=None)

    NICKEL = st.number_input('NICKEL',min_value=None, max_value=None)
    MOLYBDENUM = st.number_input('MOLYBDENUM',min_value=None, max_value=None)
    VANADIUM = st.number_input('VANADIUM',min_value=None, max_value=None)
    NITROGEN = st.number_input('NITROGEN',min_value=None, max_value=None)
    NIOBIUM = st.number_input('NIOBIUM',min_value=None, max_value=None)
    COBALT = st.number_input('COBALT',min_value=None, max_value=None)
    TUNGSTEN = st.number_input('TUNGSTEN',min_value=None, max_value=None)
    ALUMINIUM = st.number_input('ALUMINIUM',min_value=None, max_value=None)
    TITANIUM = st.number_input('TITANIUM',min_value=None, max_value=None)

    data = {'CARBON': CARBON,
            'MANGANESE': MANGANESE,
            'SILICON': SILICON,
            'CHROMIUM': CHROMIUM,

            'NICKEL': NICKEL,
            'MOLYBDENUM': MOLYBDENUM,
            'VANADIUM': VANADIUM,
            'NITROGEN': NITROGEN,
            'NIOBIUM':NIOBIUM,
            'COBALT': COBALT,
            'TUNGSTEN': TUNGSTEN,
            'ALUMINIUM': ALUMINIUM,
            'TITANIUM': TITANIUM}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()
model = pickle.load(open('finalized_model.pkl', 'rb'))
pred=model.predict(df)

# Print specified input parameters

st.write(df)
st.write('---')

st.header('Predicted Tensile Strength')
st.write(pred)
