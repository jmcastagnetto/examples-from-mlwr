# SPAM/HAM classification using caret and Naive Bayes

I am currently reading the book "Machine Learning with R"[^mlr] by Brent Lantz,
and also want to learn more about the `caret`[^caret] package, so I decided to replicate
the SPAM/HAM classification example from the chapter 4 of the book using `caret`
instead of the `e1071`[^e1071] package used in the text.

Also, instead of using as comparison the number of false positives, I decided
to use the sensitivity and specificity as criteria to evaluate the
prediction models.

Another difference is that I used the calculated models on a (different) second
dataset to test their prediction performance.

[^mlr]: [Book page at Packt](https://www.packtpub.com/big-data-and-business-intelligence/machine-learning-r)

[^caret]: [The caret package](http://caret.r-forge.r-project.org/) site

[^e1071]: [http://cran.r-project.org/web/packages/e1071/index.html](http://cran.r-project.org/web/packages/e1071/index.html)