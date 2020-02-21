
object ScalaWeek5_tp_Jean_Philippe {
  println("Welcome to the Scala worksheet")       //> Welcome to the Scala worksheet
  
//The max function and the fibonacci function exercises.

//Fibo
def fib3( n : Int) : Int = {
  def fib_tail( n: Int, a:Int, b:Int): Int = n match {
    case 0 => a
    case _ => fib_tail( n-1, b, a+b )
  }
  return fib_tail( n, 0, 1)
}                                                 //> fib3: (n: Int)Int

var x:Int = 0                                     //> x  : Int = 0

do {println(fib3(x)); x=x+1 } while (x <= 10)     //> 0
                                                  //| 1
                                                  //| 1
                                                  //| 2
                                                  //| 3
                                                  //| 5
                                                  //| 8
                                                  //| 13
                                                  //| 21
                                                  //| 34
                                                  //| 55

class TT(x:Int , y:Int){
	def this()=this(1,1)
 	def max() = if (x>y) x else y
}

val t= new TT(5,10).max()                         //> t  : Int = 10

def callmax(x:Int , y:Int):Int = new TT(x,y).max  //> callmax: (x: Int, y: Int)Int
	
callmax(5,10)                                     //> res0: Int = 10


	
//Exercise 1
//Write a function to compute factorial (5! = 5*4*3*2*1)
//Then write another function to call fact function and println few examples (i.e, 6,8,4.52)

 def factorialtr(n:Int):Int = {
   		def loop(acc:Int, n:Int): Int=
   			if (n ==0) acc
   			else loop(acc * n,n-1)
   		loop(1,n)
   }                                              //> factorialtr: (n: Int)Int
   
   factorialtr(5)                                 //> res1: Int = 120
   
	def PrintFactorial(a: Int, b :Int): Unit = {
   		for (x <- a to b) {
  			println(factorialtr(x))
		
  	}
  }                                               //> PrintFactorial: (a: Int, b: Int)Unit
PrintFactorial(1,6)                               //> 1
                                                  //| 2
                                                  //| 6
                                                  //| 24
                                                  //| 120
                                                  //| 720
 
 
//Exercise 2
//We will work with lists. Here are some codes to learn how we work with lists:
//val List = List("Alice", "John", "Dina", "Valentin")
//println(List(1))
//println(List.head)
//println(List.tail)
//for (name <- List) {println(name)}
//2-a) Then write another function to compute the factorial via reading from list. For
//instance, you will get list as (1,2,3,4,5) then multiply them together and compute the factorial.

val list = List(3,4,5,6)                          //> list  : List[Int] = List(3, 4, 5, 6)
val p = list.product                              //> p  : Int = 360

//2-b) Use the reduce method and recompute the factorial number.
def mapReduce(f: Int => Int, combine: (Int, Int) => Int,
               zero: Int)(a: Int, b: Int): Int =
  if (a > b) zero
  else combine(f(a), mapReduce(f, combine, zero)(a + 1, b))
                                                  //> mapReduce: (f: Int => Int, combine: (Int, Int) => Int, zero: Int)(a: Int, b
                                                  //| : Int)Int

def product(f: Int => Int)(a: Int, b: Int): Int =
  mapReduce(f, (x,y) => x * y, 1)(a, b)           //> product: (f: Int => Int)(a: Int, b: Int)Int

println(product(x => x)(list.head, list.last))    //> 360

//2-c) Extend the previous code to generate a list from a number (6 turns into
//list(1,2,3,4,5,6)) then compute the factorial.
def factolist(x:Int): Unit ={
	val l = 1 to x toList
	
	println(product(x => x)(l.head, l.last) , l)
 }                                                //> factolist: (x: Int)Unit
factolist(6)                                      //> (720,List(1, 2, 3, 4, 5, 6))
//Exercise 3
//Generate a list from 1 to 45 then apply .filter to compute the following results:
//Sum of the numbers divisible by 4;
val l2 = 1 to 45 toList                           //> l2  : List[Int] = List(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 1
                                                  //| 6, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 
                                                  //| 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45)
val fourmultisum = l2.filter(_ % 4 == 0).sum      //> fourmultisum  : Int = 264

//Sum of the squares of the numbers divisible by 3 and less than 20;
val l3 = 1 to 19 toList                           //> l3  : List[Int] = List(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 1
                                                  //| 6, 17, 18, 19)
def sqr(x: Int) = x * x                           //> sqr: (x: Int)Int
val sumofsquares = l3.filter(_ % 3 == 0).map(sqr).sum
                                                  //> sumofsquares  : Int = 819
  
}