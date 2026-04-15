import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

st.set_page_config(
    page_title="Bento Motors — Price Predictor",
    page_icon="",
    layout="wide"
)

RANDOM_STATE = 42
CURRENT_YEAR = 2020 #dataset only goes up to 2020

MAKES = [
    'AK', 'Abarth', 'Aixam', 'Alfa Romeo', 'Alpine', 'Ariel', 'Aston Martin',
    'Audi', 'Austin', 'BAC', 'BMW', 'Beauford', 'Bentley', 'Bristol', 'Buick',
    'CUPRA', 'Cadillac', 'Caterham', 'Chevrolet', 'Chrysler', 'Citroen',
    'Corvette', 'Custom Vehicle', 'DS AUTOMOBILES', 'Dacia', 'Daewoo',
    'Daihatsu', 'Daimler', 'Datsun', 'Dax', 'Dodge', 'Ferrari', 'Fiat', 'GMC',
    'Great Wall', 'Holden', 'Honda', 'Hummer', 'Hyundai', 'Infiniti', 'Isuzu',
    'Iveco', 'Jaguar', 'Jeep', 'Jensen', 'Kia', 'LEVC', 'Lamborghini', 'Lancia',
    'Land Rover', 'Lexus', 'Lincoln', 'London Taxis International', 'Lotus',
    'MG', 'MINI', 'Maserati', 'Maybach', 'Mazda', 'McLaren', 'Mercedes-Benz',
    'Mev', 'Mitsubishi', 'Mitsuoka', 'Morgan', 'Nissan', 'Noble', 'Opel',
    'Perodua', 'Peugeot', 'Pilgrim', 'Plymouth', 'Pontiac', 'Porsche', 'Proton',
    'Radical', 'Renault', 'Replica', 'Rolls-Royce', 'Rover', 'SEAT', 'SKODA',
    'Saab', 'Sebring', 'Smart', 'SsangYong', 'Subaru', 'Suzuki', 'TVR', 'Tesla',
    'Tiger', 'Toyota', 'Triumph', 'Ultima', 'Vauxhall', 'Volkswagen', 'Volvo',
    'Westfield', 'Zenos'
]

BODY_TYPES = ['SUV', 'Saloon', 'Hatchback', 'Convertible', 'Limousine', 'Estate',
              'MPV', 'Coupe', 'Pickup', 'Combi Van', 'Panel Van', 'Minibus',
              'Window Van', 'Car Derived Van', 'Chassis Cab']
FUEL_TYPES = ['Petrol', 'Diesel', 'Petrol Hybrid', 'Petrol Plug-in Hybrid',
              'Diesel Hybrid', 'Diesel Plug-in Hybrid', 'Electric', 'Natural Gas']
COLOURS    = ['Black', 'White', 'Grey', 'Blue', 'Silver', 'Red', 'Green', 'Orange',
              'Brown', 'Bronze', 'Purple', 'Yellow', 'Turquoise', 'Gold',
              'Multicolour', 'Burgundy', 'Pink', 'Maroon', 'Magenta', 'Navy', 'Indigo']


# Model training function
@st.cache_resource(show_spinner=False)
def train_model():
    df = pd.read_csv('adverts.csv')
    df = df.sample(n=50000, random_state=RANDOM_STATE)

    df = df.drop(columns=['public_reference', 'standard_model', 'reg_code', 'crossover_car_and_van'])
    df['car_age'] = CURRENT_YEAR - df['year_of_registration']
    df['mileage_per_year'] = df['mileage'] / df['car_age'].replace(0, 1)

    df = df[df['mileage'] < 300000]
    df = df[df['price'] >= 500]
    df = df[df['price'] <= 150000]
    df = df[df['car_age'] >= 0]
    df = df[df['car_age'] <= 30]

    df['mileage']          = df['mileage'].fillna(df['mileage'].median())
    df['car_age']          = df['car_age'].fillna(df['car_age'].median())
    df['mileage_per_year'] = df['mileage_per_year'].fillna(df['mileage_per_year'].median())
    for col in ['standard_colour', 'fuel_type', 'body_type']:
        df[col] = df[col].fillna(df[col].mode()[0])
    df = df.dropna(subset=['year_of_registration'])

    df['vehicle_condition'] = (df['vehicle_condition'] == 'NEW').astype(int)
    le = LabelEncoder()
    le.fit(MAKES)
    df = df[df['standard_make'].isin(MAKES)]
    df['standard_make'] = le.transform(df['standard_make'])
    df = pd.get_dummies(df, columns=['body_type', 'fuel_type', 'standard_colour'], drop_first=True)
    df = df.drop(columns=['year_of_registration'])

    y = np.log1p(df['price'].values)
    X = df.drop(columns=['price'])
    feature_names = list(X.columns)

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.20, random_state=RANDOM_STATE)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    model.fit(X_train_s, y_train)
    explainer = shap.TreeExplainer(model)

    return model, scaler, feature_names, explainer

# Startup loading screen 
st.title("Bento Motors — Used Car Price Predictor")
st.markdown(
    "Enter the vehicle details below to receive an estimated market price, "
    "powered by a Random Forest model trained on UK car listings."
)
st.divider()

if 'model_ready' not in st.session_state:
    with st.status("Starting up — preparing the model, please wait...", expanded=True) as status:
        st.write("Loading dataset...")
        st.write("Training Random Forest on vehicle listings...")
        st.write("Building SHAP explainer...")
        model, scaler, feature_names, explainer = train_model()
        st.session_state['model_ready'] = True
        status.update(label="Model ready!", state="complete", expanded=False)
else:
    model, scaler, feature_names, explainer = train_model()

le_make = LabelEncoder()
le_make.fit(MAKES)

def build_input_row(mileage, make, condition, car_age, body_type, fuel_type, colour):
    mileage_per_year = mileage / max(car_age, 1)
    make_encoded     = le_make.transform([make])[0]
    condition_enc    = 1 if condition == 'NEW' else 0

    row = pd.DataFrame(0, index=[0], columns=feature_names)
    row['mileage']           = mileage
    row['standard_make']     = make_encoded
    row['vehicle_condition'] = condition_enc
    row['car_age']           = car_age
    row['mileage_per_year']  = mileage_per_year

    bt_col = f'body_type_{body_type}'
    if bt_col in row.columns:
        row[bt_col] = 1

    ft_col = f'fuel_type_{fuel_type}'
    if ft_col in row.columns:
        row[ft_col] = 1

    col_col = f'standard_colour_{colour}'
    if col_col in row.columns:
        row[col_col] = 1

    return row


# UI 
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Vehicle details")
    make      = st.selectbox("Make", sorted(MAKES))
    body_type = st.selectbox("Body type", BODY_TYPES)
    fuel_type = st.selectbox("Fuel type", FUEL_TYPES)
    colour    = st.selectbox("Colour", COLOURS)

with col2:
    st.subheader("Condition & history")
    condition = st.radio("Vehicle condition", ["USED", "NEW"])
    mileage   = st.number_input("Mileage (miles)", min_value=0, max_value=300000,
                                 value=30000, step=1000)
    if condition == "NEW":
        car_age = 0
        st.slider("Car age (years)", min_value=0, max_value=30, value=0, disabled=True)
        st.caption("New vehicles have a car age of 0.")
    else:
        car_age = st.slider("Car age (years)", min_value=0, max_value=30, value=5)
with col3:
    st.subheader("Estimated price")
    predict_btn = st.button("Predict price", use_container_width=True, type="primary")

if predict_btn:
    input_row    = build_input_row(mileage, make, condition, car_age, body_type, fuel_type, colour)
    input_scaled = scaler.transform(input_row)

    log_pred   = model.predict(input_scaled)[0]
    pred_price = np.expm1(log_pred)

    st.session_state['pred_price']   = pred_price
    st.session_state['input_row']    = input_row
    st.session_state['input_scaled'] = input_scaled

if 'pred_price' in st.session_state:
    pred_price   = st.session_state['pred_price']
    input_row    = st.session_state['input_row']
    input_scaled = st.session_state['input_scaled']

    st.divider()
    col_a, col_b = st.columns(2)

    with col_a:
        st.metric("Predicted price", f"£{pred_price:,.0f}")
        low  = pred_price * 0.90
        high = pred_price * 1.10
        st.caption(f"Indicative range: £{low:,.0f} — £{high:,.0f}")

    with col_b:
        st.subheader("Why this price? (SHAP explanation)")
        with st.spinner("Generating SHAP explanation..."):
            shap_values = explainer.shap_values(input_scaled)

            if isinstance(shap_values, list):
                sv = shap_values[0][0]
                ev = explainer.expected_value[0]
            else:
                sv = shap_values[0]
                ev = explainer.expected_value[0] if hasattr(explainer.expected_value, '__len__') else explainer.expected_value

            explanation = shap.Explanation(
                values=sv,
                base_values=ev,
                data=input_row.iloc[0].values,
                feature_names=feature_names
            )

            shap.plots.waterfall(explanation, max_display=12, show=False)
            st.pyplot(plt.gcf())
            plt.close()

        st.caption(
            "Features in red push the predicted price **up**; "
            "features in blue push it **down**."
        )

st.divider()
st.markdown(
    "<small>Model: Random Forest Regressor · Trained on UK vehicle listing data · "
    "UA92 Applied AI Assessment</small>",
    unsafe_allow_html=True
)