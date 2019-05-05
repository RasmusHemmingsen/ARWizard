using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ZombieController : MonoBehaviour
{
    Animator animator;

    private void Awake()
    {
        animator = GetComponent<Animator>();
    }

    private void OnTriggerEnter(Collider other)
    {
        ZombieHit();
    }

    public void ZombieHit()
    {
        animator.ResetTrigger("Resurrect");
        animator.SetTrigger("IsHit");

        StartCoroutine(Resurrect());
    }

    private IEnumerator Resurrect()
    {
        yield return new WaitForSeconds(3f);
        animator.ResetTrigger("IsHit");
        animator.SetTrigger("Resurrect");
    }

}