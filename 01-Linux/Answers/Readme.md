# Computer vision 

## Lab 1 Linux - Git

### Sergio Alberto Galindo León - Stephannie Jimenez Gacha

1. `grep` is a Unix command that finds all lines in the text documents of an specified directory or file that contains (embedded or not) a given word, words or phrase [1]. It’s syntax is:

```
grep    [options]    word_phrase_words   directory_or_file 
```

The command is case sensitive and shows the name of the file or files containing the search terms if there is more than one output file. The output of the grep command has the following form:

```
./file1:line_containing_search_term
./file2:line_containing_search_term
.
.
.
```

Returning one line per each coincidence. The command is mainly used for searching files or documents in large directories [1].

2. The  `#!  /bin/bash` at the start of the scripts is called the shebang and it is used in order to allow the text and data files to be used as executable files under the bash shell in Linux. After the `#!` characters the route of the desired interpreter is written. In this way, the statements contained in the file are interpreted with the specific interpreter found in the given directory of the shebang [2]. It’s syntax is:

```
#!route_of_interpreter   [options_or_arguments]
```

3.Using `cat /etc/passwd|wc -l ` in the server terminal it was possible to determine that the number of users in the course servers is 38. We counted the number of lines of the etc/passwd file which contains one line per user [3] [4].

4. `cut -f 1,7 -d: /etc/passwd | tr ":" " " | sort -k2`

The `cut -f 1,7 -d: /etc/passwd` command will trim the contents of the passwd file respecting the first and seventh field  (-f 1,7) defined by the delimiter `“ : ”(-d:)`, after that, this information is passed to the tr command that replaces all `“ : ”` by spaces `“ ”` in order to have a table separated by spaces, then, this new table is passed to the sort command that organizes the lines respecting the second field (-k2) whch corresponds to the original seventh field: the shell of each user [5].

![alt text](https://github.com/steff456/IBIO4680/blob/master/01-Linux/Answers/Fig1.png)

5. Script developed for finding duplicate images based on their content

```
#!/bin/bash
cd ~

rm -r duplicate_images 2>/dev/null
mkdir duplicate_images

imagenes=$(find sipi_images -name *.tiff)

for im1 in ${imagenes[*]} 
    do 
        a=$(sha1sum $im1 | cut -f1 -d" ")
        a1=$(sha1sum $im1 | cut -f3 -d" ")

        for im2 in ${imagenes[*]}
    do
    b=$(sha1sum $im2 | cut -f1 -d" ")
    b1=$(sha1sum $im2 | cut -f3 -d" ")

    if [ "$a" == "$b" ] && [ "$a1" != "$b1" ] 
    then
        echo "find duplicate"
        echo $im1
        echo $im2
        cp $im1 duplicate_images
        cp $im2 duplicate_images
    fi 

    done 
done

```

6. The Berkeley segmentation dataset was downloaded from the web page and decompressed using

```
tar -xvzf BSR_bsds500.tgz
```

7. Using `du -h`  command in the uncompress folder directory it was found that the disk size of the uncompress dataset is 74Mb. Also, using

```
find -type f -name *.png -o -name *.jpg -o -name *.bmp -o -name *.tiff  | wc -1 
```

It was found that there is a total of 500 images in the BSR/BSDS500/data/images directory.

8. Using

```
identify $(find -type f -name *.png -o -name *.jpg -o -name *.bmp -o -name *.tiff) | cut -f3,1 -d" " |sort -k2 
```

It was found that the resolution of the images was 481x321 ( or 321x481) and their format is .jpg for all images. The identify command (of imageMagick package) is used over the output of the find command  that looks for files with .jpg, /png, .bmp or .tiff extension on their names, that corresponds to images. After that it passes this output of odentify to the cut command to obtain the first and third field (file location and  size respectively) to finally sort by size.

9. Using

```
identify $(find -type f -name *.png -o -name *.jpg -o -name *.bmp -o -name *.tiff) | cut -f3,1 -d" " |sort -k2 | grep 481x321|wc -l
```

And

```
identify $(find -type f -name *.png -o -name *.jpg -o -name *.bmp -o -name *.tiff) | cut -f3,1 -d" " |sort -k2 | grep 321x481 |wc -l
```

It was found that there are 348 images in landscape orientation (width > height) and 152 images in portrait orientation for a total of 500 images contained in the folder. The command passes the previous information obtained in the previous question to the sort, grep and wc commands to sort by resolution, find the landscape images and count the number of images with that resolution.

10. Using the crop command of imagemagick over the output of the find command the images were cropped to a size of 256x256 pixels [6]. The command used was

```
mogrify -crop 256x256+0+0 $(find -type f -name *.png -o -name *.jpg -o -name *.bmp -o -name *.tiff) 
```

![alt text](https://github.com/steff456/IBIO4680/blob/master/01-Linux/Answers/Fig2.png)
![alt text](https://github.com/steff456/IBIO4680/blob/master/01-Linux/Answers/Fig3.png)

## References
*[1] "What is grep, and how do I use it?", Kb.iu.edu, 2018. [Online]. Available: https://kb.iu.edu/d/afiy. [Accessed: 04- Feb- 2018].*

*[2] "What is the function of bash shebang?", stackExchange, 2017. .

*[3] "How To View System Users in Linux on Ubuntu | DigitalOcean", Digitalocean.com, 2018. [Online]. Available: https://www.digitalocean.com/community/tutorials/how-to-view-system-users-in-linux-on-ubuntu. [Accessed: 04- Feb- 2018].*

*[4] "IBM Knowledge Center", Ibm.com, 2018. [Online]. Available: https://www.ibm.com/support/knowledgecenter/en/ssw_aix_72/com.ibm.aix.security/passwords_etc_passwd_file.htm. [Accessed: 04- Feb- 2018].*

*[5] "Ubuntu Manpage: wc - print newline, word, and byte counts for each file", Manpages.ubuntu.com, 2018. [Online]. Available: http://manpages.ubuntu.com/manpages/trusty/man1/wc.1.html. [Accessed: 05- Feb- 2018].*

*[6] A. Thyssen, "Cutting and Bordering -- IM v6 Examples", Imagemagick.org, 2018. [Online]. Available: http://www.imagemagick.org/Usage/crop/#crop. [Accessed: 05- Feb- 2018].*