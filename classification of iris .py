{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "254055b7",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "from sklearn.datasets import load_iris"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "3bf68331",
   "metadata": {},
   "outputs": [],
   "source": [
    "df=load_iris()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "dbabbcbd",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array(['setosa', 'versicolor', 'virginica'], dtype='<U10')"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df[\"target_names\"]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "4098f5c9",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[5.1, 3.5, 1.4, 0.2],\n",
       "       [4.9, 3. , 1.4, 0.2],\n",
       "       [4.7, 3.2, 1.3, 0.2],\n",
       "       [4.6, 3.1, 1.5, 0.2],\n",
       "       [5. , 3.6, 1.4, 0.2],\n",
       "       [5.4, 3.9, 1.7, 0.4],\n",
       "       [4.6, 3.4, 1.4, 0.3],\n",
       "       [5. , 3.4, 1.5, 0.2],\n",
       "       [4.4, 2.9, 1.4, 0.2],\n",
       "       [4.9, 3.1, 1.5, 0.1],\n",
       "       [5.4, 3.7, 1.5, 0.2],\n",
       "       [4.8, 3.4, 1.6, 0.2],\n",
       "       [4.8, 3. , 1.4, 0.1],\n",
       "       [4.3, 3. , 1.1, 0.1],\n",
       "       [5.8, 4. , 1.2, 0.2],\n",
       "       [5.7, 4.4, 1.5, 0.4],\n",
       "       [5.4, 3.9, 1.3, 0.4],\n",
       "       [5.1, 3.5, 1.4, 0.3],\n",
       "       [5.7, 3.8, 1.7, 0.3],\n",
       "       [5.1, 3.8, 1.5, 0.3],\n",
       "       [5.4, 3.4, 1.7, 0.2],\n",
       "       [5.1, 3.7, 1.5, 0.4],\n",
       "       [4.6, 3.6, 1. , 0.2],\n",
       "       [5.1, 3.3, 1.7, 0.5],\n",
       "       [4.8, 3.4, 1.9, 0.2],\n",
       "       [5. , 3. , 1.6, 0.2],\n",
       "       [5. , 3.4, 1.6, 0.4],\n",
       "       [5.2, 3.5, 1.5, 0.2],\n",
       "       [5.2, 3.4, 1.4, 0.2],\n",
       "       [4.7, 3.2, 1.6, 0.2],\n",
       "       [4.8, 3.1, 1.6, 0.2],\n",
       "       [5.4, 3.4, 1.5, 0.4],\n",
       "       [5.2, 4.1, 1.5, 0.1],\n",
       "       [5.5, 4.2, 1.4, 0.2],\n",
       "       [4.9, 3.1, 1.5, 0.2],\n",
       "       [5. , 3.2, 1.2, 0.2],\n",
       "       [5.5, 3.5, 1.3, 0.2],\n",
       "       [4.9, 3.6, 1.4, 0.1],\n",
       "       [4.4, 3. , 1.3, 0.2],\n",
       "       [5.1, 3.4, 1.5, 0.2],\n",
       "       [5. , 3.5, 1.3, 0.3],\n",
       "       [4.5, 2.3, 1.3, 0.3],\n",
       "       [4.4, 3.2, 1.3, 0.2],\n",
       "       [5. , 3.5, 1.6, 0.6],\n",
       "       [5.1, 3.8, 1.9, 0.4],\n",
       "       [4.8, 3. , 1.4, 0.3],\n",
       "       [5.1, 3.8, 1.6, 0.2],\n",
       "       [4.6, 3.2, 1.4, 0.2],\n",
       "       [5.3, 3.7, 1.5, 0.2],\n",
       "       [5. , 3.3, 1.4, 0.2],\n",
       "       [7. , 3.2, 4.7, 1.4],\n",
       "       [6.4, 3.2, 4.5, 1.5],\n",
       "       [6.9, 3.1, 4.9, 1.5],\n",
       "       [5.5, 2.3, 4. , 1.3],\n",
       "       [6.5, 2.8, 4.6, 1.5],\n",
       "       [5.7, 2.8, 4.5, 1.3],\n",
       "       [6.3, 3.3, 4.7, 1.6],\n",
       "       [4.9, 2.4, 3.3, 1. ],\n",
       "       [6.6, 2.9, 4.6, 1.3],\n",
       "       [5.2, 2.7, 3.9, 1.4],\n",
       "       [5. , 2. , 3.5, 1. ],\n",
       "       [5.9, 3. , 4.2, 1.5],\n",
       "       [6. , 2.2, 4. , 1. ],\n",
       "       [6.1, 2.9, 4.7, 1.4],\n",
       "       [5.6, 2.9, 3.6, 1.3],\n",
       "       [6.7, 3.1, 4.4, 1.4],\n",
       "       [5.6, 3. , 4.5, 1.5],\n",
       "       [5.8, 2.7, 4.1, 1. ],\n",
       "       [6.2, 2.2, 4.5, 1.5],\n",
       "       [5.6, 2.5, 3.9, 1.1],\n",
       "       [5.9, 3.2, 4.8, 1.8],\n",
       "       [6.1, 2.8, 4. , 1.3],\n",
       "       [6.3, 2.5, 4.9, 1.5],\n",
       "       [6.1, 2.8, 4.7, 1.2],\n",
       "       [6.4, 2.9, 4.3, 1.3],\n",
       "       [6.6, 3. , 4.4, 1.4],\n",
       "       [6.8, 2.8, 4.8, 1.4],\n",
       "       [6.7, 3. , 5. , 1.7],\n",
       "       [6. , 2.9, 4.5, 1.5],\n",
       "       [5.7, 2.6, 3.5, 1. ],\n",
       "       [5.5, 2.4, 3.8, 1.1],\n",
       "       [5.5, 2.4, 3.7, 1. ],\n",
       "       [5.8, 2.7, 3.9, 1.2],\n",
       "       [6. , 2.7, 5.1, 1.6],\n",
       "       [5.4, 3. , 4.5, 1.5],\n",
       "       [6. , 3.4, 4.5, 1.6],\n",
       "       [6.7, 3.1, 4.7, 1.5],\n",
       "       [6.3, 2.3, 4.4, 1.3],\n",
       "       [5.6, 3. , 4.1, 1.3],\n",
       "       [5.5, 2.5, 4. , 1.3],\n",
       "       [5.5, 2.6, 4.4, 1.2],\n",
       "       [6.1, 3. , 4.6, 1.4],\n",
       "       [5.8, 2.6, 4. , 1.2],\n",
       "       [5. , 2.3, 3.3, 1. ],\n",
       "       [5.6, 2.7, 4.2, 1.3],\n",
       "       [5.7, 3. , 4.2, 1.2],\n",
       "       [5.7, 2.9, 4.2, 1.3],\n",
       "       [6.2, 2.9, 4.3, 1.3],\n",
       "       [5.1, 2.5, 3. , 1.1],\n",
       "       [5.7, 2.8, 4.1, 1.3],\n",
       "       [6.3, 3.3, 6. , 2.5],\n",
       "       [5.8, 2.7, 5.1, 1.9],\n",
       "       [7.1, 3. , 5.9, 2.1],\n",
       "       [6.3, 2.9, 5.6, 1.8],\n",
       "       [6.5, 3. , 5.8, 2.2],\n",
       "       [7.6, 3. , 6.6, 2.1],\n",
       "       [4.9, 2.5, 4.5, 1.7],\n",
       "       [7.3, 2.9, 6.3, 1.8],\n",
       "       [6.7, 2.5, 5.8, 1.8],\n",
       "       [7.2, 3.6, 6.1, 2.5],\n",
       "       [6.5, 3.2, 5.1, 2. ],\n",
       "       [6.4, 2.7, 5.3, 1.9],\n",
       "       [6.8, 3. , 5.5, 2.1],\n",
       "       [5.7, 2.5, 5. , 2. ],\n",
       "       [5.8, 2.8, 5.1, 2.4],\n",
       "       [6.4, 3.2, 5.3, 2.3],\n",
       "       [6.5, 3. , 5.5, 1.8],\n",
       "       [7.7, 3.8, 6.7, 2.2],\n",
       "       [7.7, 2.6, 6.9, 2.3],\n",
       "       [6. , 2.2, 5. , 1.5],\n",
       "       [6.9, 3.2, 5.7, 2.3],\n",
       "       [5.6, 2.8, 4.9, 2. ],\n",
       "       [7.7, 2.8, 6.7, 2. ],\n",
       "       [6.3, 2.7, 4.9, 1.8],\n",
       "       [6.7, 3.3, 5.7, 2.1],\n",
       "       [7.2, 3.2, 6. , 1.8],\n",
       "       [6.2, 2.8, 4.8, 1.8],\n",
       "       [6.1, 3. , 4.9, 1.8],\n",
       "       [6.4, 2.8, 5.6, 2.1],\n",
       "       [7.2, 3. , 5.8, 1.6],\n",
       "       [7.4, 2.8, 6.1, 1.9],\n",
       "       [7.9, 3.8, 6.4, 2. ],\n",
       "       [6.4, 2.8, 5.6, 2.2],\n",
       "       [6.3, 2.8, 5.1, 1.5],\n",
       "       [6.1, 2.6, 5.6, 1.4],\n",
       "       [7.7, 3. , 6.1, 2.3],\n",
       "       [6.3, 3.4, 5.6, 2.4],\n",
       "       [6.4, 3.1, 5.5, 1.8],\n",
       "       [6. , 3. , 4.8, 1.8],\n",
       "       [6.9, 3.1, 5.4, 2.1],\n",
       "       [6.7, 3.1, 5.6, 2.4],\n",
       "       [6.9, 3.1, 5.1, 2.3],\n",
       "       [5.8, 2.7, 5.1, 1.9],\n",
       "       [6.8, 3.2, 5.9, 2.3],\n",
       "       [6.7, 3.3, 5.7, 2.5],\n",
       "       [6.7, 3. , 5.2, 2.3],\n",
       "       [6.3, 2.5, 5. , 1.9],\n",
       "       [6.5, 3. , 5.2, 2. ],\n",
       "       [6.2, 3.4, 5.4, 2.3],\n",
       "       [5.9, 3. , 5.1, 1.8]])"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "073a4b1f",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import train_test_split"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "6b677a9f",
   "metadata": {},
   "outputs": [],
   "source": [
    "x=df[\"data\"]\n",
    "y=df[\"target\"]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "f212fae3",
   "metadata": {},
   "outputs": [],
   "source": [
    "x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=32)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "77efcb8a",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.linear_model import LogisticRegression"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "d3abd049",
   "metadata": {},
   "outputs": [],
   "source": [
    "lr=LogisticRegression()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "f9e22d74",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "LogisticRegression()"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "lr.fit(x_train,y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "072830f7",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "1.0"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "lr.score(x_test,y_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "88166648",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}