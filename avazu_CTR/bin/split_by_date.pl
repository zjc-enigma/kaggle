#!/usr/bin/perl
my $train_file="../data/train";
my $write_dir="../data";


my $current_date = "";
my $content = "";

open(RFD, $train_file);
while(<RFD>){
    my $line = $_;
    chomp;

    my @arr = split/,/;
    my $timestamp = $arr[2];
    $timestamp =~ m/(1410\d{2})/;

    my $day = $1;
    if($day ne $current_date) {

        unless ($current_date eq ""){
            open(WFD, ">>$write_dir/$current_date");
            print WFD $content;
            close(WFD);
        }
        $content = $line;
        $current_date = $day;
    }

    else {
        $content .= $line;
    }
}
close(RFD);
