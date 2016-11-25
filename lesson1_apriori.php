<?php

require_once('library/Apriori/lib/Apriori.class.php');

//variables
$minSupp  = 5;                  //minimal support
$minConf  = 75;                 //minimal confidence
$type     = Apriori::SRC_PLAIN; //data type
$recomFor = 'beer';             //recommendation for

$data = 'lesson1_dataset.txt';

$apri = new Apriori($type, $data, $minSupp, $minConf);
$apri->displayTransactions()
    ->solve();

$apri->generateRules()
    ->displayRules()
    ->displayRecommendations($recomFor);

unset($apri);

?>