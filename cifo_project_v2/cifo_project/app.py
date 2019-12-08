
from flask import Flask, render_template, request, url_for, redirect, session

#--------------------------------------------------------------------------------------------------
# Global
#--------------------------------------------------------------------------------------------------
# Flask App
app = Flask(__name__)


#--------------------------------------------------------------------------------------------------
# Route: Index
#--------------------------------------------------------------------------------------------------
@app.route('/index')
@app.route('/')
def index():
    
    print("Hello World")

#--------------------------------------------------------------------------------------------------
# MAIN
#--------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    #app_dash.run_server(debug = True)
    app.run(debug = True) 