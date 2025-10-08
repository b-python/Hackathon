import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import webbrowser
from flask import Flask, send_from_directory

import os
import logging
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from flask import Flask, request, redirect, url_for, flash, session, render_template
from flask_mail import Mail, Message
import re

app = Flask(__name__, template_folder='../web')  # Specify the templates folder
app.secret_key = os.urandom(24)  # Use a secure random key for sessions

# Configure Flask-Mail
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'alain50@gmail.com'  
app.config['MAIL_PASSWORD'] = 'apppassword'  
mail = Mail(app)

# Set up logging
logging.basicConfig(filename='logs/contact_form.log', level=logging.INFO)

# CSRF token generation
@app.before_request
def before_request():
    if 'csrf_token' not in session:
        session['csrf_token'] = os.urandom(32).hex()

# Load datasets
child_health = pd.read_csv('../data/child_health_indicators.csv')
family_socioeconomic = pd.read_csv('../data/family_socioeconomic.csv')
healthcare_access = pd.read_csv('../data/healthcare_access.csv')
nutritional_practices = pd.read_csv('../data/nutritional_practices.csv')
psychosocial_factors = pd.read_csv('../data/psychosocial_factors.csv')
maternal_health_factors = pd.read_csv('../data/maternal_health_factors.csv')
child_development = pd.read_csv('../data/child_development.csv')
food_security_access = pd.read_csv('../data/food_security_access.csv')
health_monitoring = pd.read_csv('../data/health_monitoring.csv')
community_support = pd.read_csv('../data/community_support.csv')

# Contact form route
@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        # Honeypot field
        honeypot = request.form.get('honeypot')
        if honeypot:
            return "Spam detected.", 400

        # Input sanitization
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip()
        phone = request.form.get('phone', '').strip()
        subject = request.form.get('subject', '').strip()
        message = request.form.get('message', '').strip()
        csrf_token = request.form.get('csrf_token', '')

        # Validate CSRF token
        if not compare_csrf_token(session['csrf_token'], csrf_token):
            flash("CSRF token validation failed.", "danger")
            return redirect(url_for('contact'))

        # Validate input
        if not validate_input(name, email, phone, subject, message):
            flash("Invalid input, please check your entries.", "danger")
            return redirect(url_for('contact'))

        # Log submission
        logging.info(f"Submission from {request.remote_addr}: {name}, {email}")

        try:
            # Send email
            send_email(name, email, phone, subject, message)
            flash("Message sent successfully! A confirmation email has been sent to you.", "success")
            return redirect(url_for('contact'))
        except Exception as e:
            logging.error(f"Error sending email: {e}")
            flash("Message could not be sent. Please try again later.", "danger")

    return render_template('contact.html')

def compare_csrf_token(session_token, form_token):
    return session_token == form_token

def validate_input(name, email, phone, subject, message):
    # Validate name
    if not re.match(r'^[A-Za-z\s]+$', name):
        return False
    # Validate email
    if not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', email):
        return False
    # Validate phone
    if not re.match(r'^\d{10}$', phone):
        return False
    # Validate subject
    if subject not in ["General Inquiry", "Feedback", "Technical Support", "Other"]:
        return False
    # Validate message
    if len(message) > 500:
        return False
    return True

def send_email(name, email, phone, subject, message):
    # Send email to admin
    msg = Message('New Contact Form Submission', sender=email, recipients=['alain50@gmail.com'])
    msg.body = f"Name: {name}\nEmail: {email}\nPhone: {phone}\nSubject: {subject}\n\nMessage:\n{message}"
    mail.send(msg)

    # Send confirmation email to user
    confirmation_msg = Message('Thank You for Your Submission', sender='alain50@gmail.com', recipients=[email])
    confirmation_msg.body = f"Dear {name},\n\nThank you for reaching out! We have received your message:\n\n{message}\n\nWe will get back to you shortly.\n\nBest regards,\nUR-CAVM Team"
    mail.send(confirmation_msg)



# Load datasets
child_health = pd.read_csv('../data/child_health_indicators.csv')
family_socioeconomic = pd.read_csv('../data/family_socioeconomic.csv')
healthcare_access = pd.read_csv('../data/healthcare_access.csv')
nutritional_practices = pd.read_csv('../data/nutritional_practices.csv')
psychosocial_factors = pd.read_csv('../data/psychosocial_factors.csv')
maternal_health_factors = pd.read_csv('../data/maternal_health_factors.csv')
child_development = pd.read_csv('../data/child_development.csv')
food_security_access = pd.read_csv('../data/food_security_access.csv')
health_monitoring = pd.read_csv('../data/health_monitoring.csv')
community_support = pd.read_csv('../data/community_support.csv')

# Function to visualize Height vs Weight with histograms and average weight line
def visualize_height_weight(data):
    fig = plt.figure(figsize=(12, 8))
    
    # Scatter plot
    ax1 = fig.add_subplot(211)
    sns.scatterplot(data=data, x='Height (cm)', y='Weight (kg)', hue='Dietary Diversity Score', palette='viridis', ax=ax1, alpha=0.6)
    
    # Calculate average weight per height interval
    height_bins = range(80, 130, 5)  # Define height bins from 80 to 130 cm
    data['Height Bin'] = pd.cut(data['Height (cm)'], bins=height_bins, right=False)
    average_weights = data.groupby('Height Bin')['Weight (kg)'].mean().reset_index()

    # Extract bin midpoints for plotting
    average_weights['Height (cm)'] = average_weights['Height Bin'].apply(lambda x: x.mid)

    # Plotting the average weight line
    ax1.plot(average_weights['Height (cm)'], average_weights['Weight (kg)'], marker='o', color='red', label='Average Weight Line')
    
    ax1.set_title('Height vs Weight of Children', fontsize=16)
    ax1.set_xlabel('Height (cm)', fontsize=14)
    ax1.set_ylabel('Weight (kg)', fontsize=14)
    ax1.legend(title='Dietary Diversity Score')
    ax1.grid(True)

    # Histograms for Height and Weight
    ax2 = fig.add_subplot(212)
    sns.histplot(data=data, x='Height (cm)', bins=10, kde=True, color='blue', alpha=0.5, ax=ax2, stat="density")
    sns.histplot(data=data, y='Weight (kg)', bins=10, kde=True, color='orange', alpha=0.5, ax=ax2, stat="density", orientation='horizontal')
    
    ax2.set_title('Height and Weight Distributions', fontsize=16)
    ax2.set_xlabel('Density of Height (cm)', fontsize=14)
    ax2.set_ylabel('Density of Weight (kg)', fontsize=14)
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('../output/height_weight_scatter.png')
    plt.close()

# Function to visualize histogram with normal curve
def visualize_histogram_with_kde(data, column, title, xlabel, output_file):
    plt.figure(figsize=(10, 6))
    sns.histplot(data[column], kde=True, stat="density", linewidth=0, color='skyblue')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Density')
    plt.grid(True)
    plt.savefig(output_file)
    plt.close()

# Analyze Child Health Indicators
def analyze_child_health():
    visualize_height_weight(child_health)
    average_weight = child_health['Weight (kg)'].mean()
    print(f'Average Weight of Children: {average_weight:.2f} kg')
    
    # Visualize weight distribution
    visualize_histogram_with_kde(child_health, 'Weight (kg)', 'Weight Distribution of Children', 'Weight (kg)', '../output/weight_distribution.png')
    
    return average_weight

# Analyze Family Socioeconomic Factors
def analyze_family_socioeconomic():
    plt.figure(figsize=(10, 6))
    sns.countplot(data=family_socioeconomic, x='Wealth Quintile', hue='Wealth Quintile', palette='pastel', legend=False)
    plt.title('Distribution of Wealth Quintiles')
    plt.xlabel('Wealth Quintile')
    plt.ylabel('Number of Children')
    plt.grid(True)
    plt.savefig('../output/wealth_quintile_distribution.png')
    plt.close()
    print("This graph shows the distribution of children across different wealth quintiles, indicating the economic status of families.")

# Analyze Healthcare Access
def analyze_healthcare_access():
    plt.figure(figsize=(10, 6))
    sns.barplot(data=healthcare_access, x='Quality of Healthcare', y='Distance to Healthcare (km)', hue='Quality of Healthcare', palette='coolwarm', legend=False)
    plt.title('Distance to Healthcare by Quality')
    plt.xlabel('Quality of Healthcare')
    plt.ylabel('Distance (km)')
    plt.grid(True)
    plt.savefig('../output/distance_healthcare_quality.png')
    plt.close()
    print("This bar graph illustrates the distance to healthcare facilities based on the perceived quality, highlighting accessibility issues.")

# Analyze Nutritional Practices
def analyze_nutritional_practices():
    plt.figure(figsize=(10, 6))
    sns.countplot(data=nutritional_practices, x='Traditional Feeding Practices', hue='Traditional Feeding Practices', palette='Set2', legend=False)
    plt.title('Distribution of Traditional Feeding Practices')
    plt.xlabel('Traditional Feeding Practices')
    plt.ylabel('Number of Children')
    plt.grid(True)
    plt.savefig('../output/traditional_feeding_practices.png')
    plt.close()
    print("This chart displays the prevalence of traditional feeding practices among children, showing cultural influences on nutrition.")

# Analyze Psychosocial Factors
def analyze_psychosocial_factors():
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=psychosocial_factors, x='Parental Stress Levels', y='Household Debt Levels', palette='muted')
    plt.title('Household Debt Levels by Parental Stress Levels')
    plt.xlabel('Parental Stress Levels')
    plt.ylabel('Household Debt Levels')
    plt.grid(True)
    plt.savefig('../output/debt_stress_levels.png')
    plt.close()
    print("This boxplot shows the relationship between parental stress levels and household debt, indicating potential financial stressors.")

# Analyze Maternal Health Factors
def analyze_maternal_health():
    plt.figure(figsize=(10, 6))
    sns.barplot(data=maternal_health_factors, x='Maternal Education Level', y='Maternal Weight (kg)', hue='Maternal Education Level', palette='Blues', legend=False)
    plt.title('Maternal Weight by Education Level')
    plt.xlabel('Maternal Education Level')
    plt.ylabel('Maternal Weight (kg)')
    plt.grid(True)
    plt.savefig('../output/maternal_weight_education.png')
    plt.close()

    # Visualize maternal weight distribution
    visualize_histogram_with_kde(maternal_health_factors, 'Maternal Weight (kg)', 'Maternal Weight Distribution', 'Maternal Weight (kg)', '../output/maternal_weight_distribution.png')

    # Analyze University Education Impact
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=maternal_health_factors, x='University', y='Maternal Weight (kg)', palette='pastel')
    plt.title('Maternal Weight by University Education')
    plt.xlabel('University Education')
    plt.ylabel('Maternal Weight (kg)')
    plt.grid(True)
    plt.savefig('../output/maternal_weight_university.png')
    plt.close()
    print("This analysis examines the impact of university education on maternal weight, highlighting differences based on education level.")

# Analyze Child Development
def analyze_child_development():
    plt.figure(figsize=(10, 6))
    sns.countplot(data=child_development, x='Cognitive Development Score', hue='Cognitive Development Score', palette='Paired', legend=False)
    plt.title('Cognitive Development Score Distribution')
    plt.xlabel('Cognitive Development Score')
    plt.ylabel('Number of Children')
    plt.grid(True)
    plt.savefig('../output/cognitive_development_distribution.png')
    plt.close()
    print("This chart represents the distribution of cognitive development scores among children, indicating developmental milestones.")

# Analyze Food Security Access
def analyze_food_security_access():
    plt.figure(figsize=(10, 6))
    sns.countplot(data=food_security_access, x='Food Security Status', hue='Food Security Status', palette='pastel', legend=False)
    plt.title('Food Security Status Distribution')
    plt.xlabel('Food Security Status')
    plt.ylabel('Number of Households')
    plt.grid(True)
    plt.savefig('../output/food_security_status_distribution.png')
    plt.close()
    print("This graph illustrates the distribution of food security status among households, highlighting food access issues.")

# Analyze Healthcare Monitoring
def analyze_health_monitoring():
    plt.figure(figsize=(10, 6))
    sns.countplot(data=health_monitoring, x='Growth Monitoring Frequency', hue='Health Check-up Outcomes', palette='Set2', legend=True)
    plt.title('Growth Monitoring Frequency vs Health Check-up Outcomes')
    plt.xlabel('Growth Monitoring Frequency')
    plt.ylabel('Number of Children')
    plt.grid(True)
    plt.savefig('../output/growth_monitoring_outcomes.png')
    plt.close()
    print("This chart shows the relationship between growth monitoring frequency and health check-up outcomes, indicating the effectiveness of monitoring.")

# Analyze Community Support
def analyze_community_support():
    plt.figure(figsize=(10, 6))
    sns.countplot(data=community_support, x='Community Support Availability', hue='Access to Parenting Resources', palette='Set2', legend=True)
    plt.title('Community Support vs Access to Parenting Resources')
    plt.xlabel('Community Support Availability')
    plt.ylabel('Number of Parents')
    plt.grid(True)
    plt.savefig('../output/community_support_resources.png')
    plt.close()
    print("This graph shows how community support impacts access to parenting resources, illustrating the importance of community networks.")

# Predictive Modeling: Predicting Weight based on Height and Age
def predictive_modeling(data):
    X = data[['Height (cm)', 'Age (months)']]
    y = data['Weight (kg)']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    # Save predictions
    prediction_results = pd.DataFrame({'Actual Weight': y_test, 'Predicted Weight': predictions})
    prediction_results.to_csv('../output/predictions.csv', index=False)

    return prediction_results

# Run analyses
average_weight = analyze_child_health()
analyze_family_socioeconomic()
analyze_healthcare_access()
analyze_nutritional_practices()
analyze_psychosocial_factors()
analyze_maternal_health()
analyze_child_development()
analyze_food_security_access()
analyze_health_monitoring()
analyze_community_support()

# General predictive modeling
predictions = predictive_modeling(child_health)

# Write average weight to HTML
html_file_path = '../web/index.html'  # Adjust path if needed
with open(html_file_path, 'r') as file:
    html_content = file.readlines()

# Update the average weight in the HTML content
for i, line in enumerate(html_content):
    if 'Average Weight of Children:' in line:
        html_content[i] = f'<p>Average Weight of Children: {average_weight:.2f} kg</p>\n'

# Write the updated content back to index.html
with open(html_file_path, 'w') as file:
    file.writelines(html_content)

# Function to serve static files (images from the output directory)
@app.route('/output/<path:filename>')
def serve_output(filename):
    return send_from_directory('../output', filename)

# Serve static files (CSS)
@app.route('/styles/<path:filename>')
def serve_styles(filename):
    return send_from_directory(os.path.join('web', 'styles'), filename)

# Function to serve static images (from the images directory)
@app.route('/images/<path:filename>')
def serve_images(filename):
    return send_from_directory('../images', filename)

# Function to serve index.html file
@app.route('/')
def serve_index():
    return send_from_directory('../web', 'index.html')


@app.route('/predictions')
def serve_predictions():
    return send_from_directory('../web', 'predictions.html')

# Serve hidden hunger page
@app.route('/hunger')
def serve_hunger():
    return send_from_directory('../web', 'hunger.html')  # Ensure hunger.html exists in the web directory

# Serve About Us page
@app.route('/about_us')
def serve_about_us():
    return send_from_directory('../web', 'about_us.html')  # Ensure about_us.html exists in the web directory

# Serve Contact Us page
@app.route('/contact')
def serve_contact():
    return send_from_directory('../web', 'contact.html')


# Start the Flask server
if __name__ == '__main__':
    # Open the web browser automatically
    webbrowser.open("http://127.0.0.1:5000/")
    app.run(debug=True, host='127.0.0.1', port=5000) 