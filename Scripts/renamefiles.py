{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Pythono3 code to rename multiple  \n",
    "# files in a directory or folder \n",
    "  \n",
    "# importing os module \n",
    "import os \n",
    "  \n",
    "# Function to rename multiple files \n",
    "def main(): \n",
    "    i = 0\n",
    "      \n",
    "    for filename in os.listdir(\"finish/\"): \n",
    "        dst =\"Chess\" + str(i) + \".jpg\"\n",
    "        src ='finish/'+ filename \n",
    "        dst ='finish/'+ dst \n",
    "          \n",
    "        # rename() function will \n",
    "        # rename all the files \n",
    "        os.rename(src, dst) \n",
    "        i += 1\n",
    "  \n",
    "# Driver Code \n",
    "if __name__ == '__main__': \n",
    "      \n",
    "    # Calling main() function \n",
    "    main() "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
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
   "version": "3.7.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
