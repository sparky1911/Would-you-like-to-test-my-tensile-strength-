import pandas as pd
import numpy as np
import joblib
import streamlit
import pickle

nnmodel = pickle.load(open('finalized_model.pkl', 'rb'))

def xgbr_prediction(CARBON,MANGANESE,SILICON,CHROMIUM,NICKEL,MOLYBDENUM,VANADIUM,NITROGEN,NIOBIUM,COBALT,TUNGSTEN,ALUMINIUM,TITANIUM):
    pred_arr=np.array([CARBON,MANGANESE,SILICON,CHROMIUM,NICKEL,MOLYBDENUM,VANADIUM,NITROGEN,NIOBIUM,COBALT,TUNGSTEN,ALUMINIUM,TITANIUM])
    preds=pred_arr.reshape(1,-1)
    preds=preds.astype(int)
    model_prediction=nnmodel.predict(preds)
    return model_prediction

def run():
    streamlit.title("Tensile strength of steel")
    html_temp="""
    """
    streamlit.markdown(html_temp)

    CARBON = streamlit.text_input('CARBON')
    MANGANESE = streamlit.text_input('MANGANESE')
    SILICON = streamlit.text_input('SILICON')
    CHROMIUM = streamlit.text_input('CHROMIUM')

    NICKEL = streamlit.text_input('NICKEL')
    MOLYBDENUM = streamlit.text_input('MOLYBDENUM')
    VANADIUM = streamlit.text_input('VANADIUM')
    NITROGEN = streamlit.text_input('NITROGEN')
    NIOBIUM = streamlit.text_input('NIOBIUM')
    COBALT = streamlit.text_input('COBALT')
    TUNGSTEN = streamlit.text_input('TUNGSTEN')
    ALUMINIUM = streamlit.text_input('ALUMINIUM')
    TITANIUM = streamlit.text_input('TITANIUM')

    prediction=""
    if streamlit.button("predict"):
        prediction=xgbr_prediction(CARBON,MANGANESE,SILICON,CHROMIUM,NICKEL,MOLYBDENUM,VANADIUM,NITROGEN,NIOBIUM,COBALT,TUNGSTEN,ALUMINIUM,TITANIUM)


        streamlit.success("Tensile strength is : {}".format(prediction))

if __name__=='__main__':
    run()
