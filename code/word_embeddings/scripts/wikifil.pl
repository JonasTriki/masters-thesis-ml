#!/usr/bin/perl

# Program to filter Wikipedia XML dumps to "clean" text consisting only of lowercase
# letters (a-z, converted from A-Z), and spaces (never consecutive).  
# All other characters are converted to spaces.  Only text which normally appears 
# in the web browser is displayed.  Tables are removed.  Image captions are 
# preserved.  Links are converted to normal text.

# Written by Matt Mahoney, June 10, 2006.  This program is released to the public domain.

# Modified by Jonas Folkvord Triki, September 04, 2020.
# Downloaded from http://mattmahoney.net/dc/textdata, Appendix A, on September 04, 2020.

$/=">";                     # input record separator
while (<>) {
  if (/<text /) {$text=1;}  # remove all but between <text> ... </text>
  if (/#redirect/i) {$text=0;}  # remove #REDIRECT
  if ($text) {

    # Remove any text not normally visible
    if (/<\/text>/) {$text=0;}
    s/<.*>//;               # remove xml tags
    s/&amp;/&/g;            # decode URL encoded chars
    s/&lt;/</g;
    s/&gt;/>/g;
    s/<ref[^<]*<\/ref>//g;  # remove references <ref...> ... </ref>
    s/<[^>]*>//g;           # remove xhtml tags
    s/\[http:[^] ]*/[/g;    # remove normal url, preserve visible text
    s/\|thumb//ig;          # remove images links, preserve caption
    s/\|left//ig;
    s/\|right//ig;
    s/\|\d+px//ig;
    s/\[\[image:[^\[\]]*\|//ig;
    s/\[\[category:([^|\]]*)[^]]*\]\]/[[$1]]/ig;  # show categories without markup
    s/\[\[[a-z\-]*:[^\]]*\]\]//g;  # remove links to other languages
    s/\[\[[^\|\]]*\|/[[/g;  # remove wiki url, preserve visible text
    s/\{\{[^\}]*\}\}//g;         # remove {{icons}} and {tables}
    s/\{[^\}]*\}//g;
    s/\[//g;                # remove [ and ]
    s/\]//g;
    s/&[^;]*;/ /g;          # remove URL encoded chars

    # convert to lowercase letters and spaces
    $_=" $_ ";
    tr/A-Z/a-z/;

    # convert all characters except a-z and '.' to space
    tr/a-z\./ /cs;
    chop;
    print $_;
  }
}