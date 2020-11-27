# An application supporting preliminary dermatological diagnosis.

Project of web application using Django framework.
Project based on simple image processing using OpenCV library and machine learning using scikit-learn.

## Installation

1. Be sure, you have [Python 3.7](https://www.python.org/downloads/release/python-374/) and [pip](https://pip.pypa.io/en/stable/) installed.
2. Clone repository.
3. Create virtual enviroment, preferably in same folder where derm_site directory exists. Use:

```bash
python -m venv my_venv_name
```

4. Run your virtual enviroment by running activate file (or activate.bat on Windows):

Unix
```bash
source/bin/activate
```

Windows
```bash
source\Scripts\activate.bat
```

5. Install all required packages using requirements.txt file:

```bash
pip install -r requirements.txt
```

6. Create file called ,,.env'' in derm_site/derm_site folder. Be sure, it's not saved as .txt file.
In .env file you need to define SECRET_KEY variable as follows:

```bash
SECRET_KET=<your key>
```

In order to generate unique secret key you can use [this](https://miniwebtool.com/django-secret-key-generator/).

7. Use makemigrations and migrate command to create database files:

```bash
python manage.py makemigrations
python manage.py migrate
```

8. Create admin account for website:

```bash
python manage.py createsuperuser
```

9. Run server:

```bash
python manage.py runserver
```


## Functionality

Application have two main functions:

1. Simple automatic image processing - function performs automatic contour detection and computes color and assymetry features of uploaded image.
2. Automatic melanoma detection - function uses machine learning methods to detect malignant skin nevus.




## References
Most of methods was implemented according to:

Alcon, JosÉ & Ciuhu, Calina & Kate, Warner & Heinrich, Adrienne & Uzunbajakava, Natallia & Krekels, Gertruud & Siem, Denny & Haan, Gerard. (2009). Automatic Imaging System With Decision Support for Inspection of Pigmented Skin Lesions and Melanoma Diagnosis. Selected Topics in Signal Processing, IEEE Journal of. 3. 14 - 25. 10.1109/JSTSP.2008.2011156.
